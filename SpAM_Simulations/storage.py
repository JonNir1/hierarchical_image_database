"""Compact, streamable on-disk storage for MDS-sweep results.

The pre-refactor notebook appended every result (including a full ``confdist`` vector of
``N(N-1)/2`` float64 values) to a single growing ``mds_results.pkl`` via repeated
``pickle.dump``. For the full sweep that file reached multiple GB and had to be read back
in its entirety. ``ResultStore`` replaces that with a small directory:

* ``store_info.json`` - the fixed confdist length and dtype.
* ``meta.csv``        - one human-readable row per result (parameters + niter/stress/status
                        + the row index of its confdist, or -1 if none).
* ``confdists.f32``   - a flat little-endian float32 binary; row ``i`` is one confdist vector.

Records are appended incrementally (crash-safe, flushed per write) and read back lazily
(``confdists.f32`` is memory-mapped, so individual vectors are paged in on demand instead of
loading everything into RAM). Storing confdists as float32 also halves them versus the
original float64 pickle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

_INFO_FILE = "store_info.json"
_META_FILE = "meta.csv"
_CONFDIST_FILE = "confdists.f32"
_CONFDIST_DTYPE = np.float32
_ROW_COLUMN = "confdist_row"
_FORMAT_VERSION = 1


class ResultStore:
    """Append-only, lazily-read store for sweep results keyed by arbitrary metadata."""

    def __init__(self, path: str | Path, confdist_len: int, meta_columns: List[str]):
        self.path = Path(path)
        self.confdist_len = int(confdist_len)
        self._meta_columns = list(meta_columns)
        self._n_confdists = 0
        self._confdist_fh = None  # lazily opened append handle

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, path: str | Path, confdist_len: int, meta_columns: List[str],
               overwrite: bool = False) -> "ResultStore":
        """Create a new store directory. ``meta_columns`` are the per-record fields
        (e.g. the experiment parameters plus 'rep', 'ndim'); status fields may also appear."""
        path = Path(path)
        info_path = path / _INFO_FILE
        if info_path.exists() and not overwrite:
            raise FileExistsError(f"A store already exists at {path}; pass overwrite=True or use open().")
        path.mkdir(parents=True, exist_ok=True)
        if _ROW_COLUMN in meta_columns:
            raise ValueError(f"'{_ROW_COLUMN}' is reserved and must not be a metadata column")
        info = {
            "format_version": _FORMAT_VERSION,
            "confdist_len": int(confdist_len),
            "confdist_dtype": np.dtype(_CONFDIST_DTYPE).str,
            "meta_columns": list(meta_columns),
        }
        info_path.write_text(json.dumps(info, indent=2))
        # (re)initialise data files
        (path / _CONFDIST_FILE).write_bytes(b"")
        with open(path / _META_FILE, "w", newline="") as f:
            csv.writer(f).writerow(list(meta_columns) + [_ROW_COLUMN])
        return cls(path, confdist_len, meta_columns)

    @classmethod
    def open(cls, path: str | Path) -> "ResultStore":
        """Open an existing store for reading and/or further appends."""
        path = Path(path)
        info = json.loads((path / _INFO_FILE).read_text())
        store = cls(path, info["confdist_len"], info["meta_columns"])
        confdist_bytes = (path / _CONFDIST_FILE).stat().st_size
        itemsize = np.dtype(_CONFDIST_DTYPE).itemsize
        store._n_confdists = confdist_bytes // (store.confdist_len * itemsize)
        return store

    # ------------------------------------------------------------------ writing
    def append(self, meta: Dict[str, Any], confdist: Optional[np.ndarray] = None) -> None:
        """Append one result. ``confdist`` may be None (e.g. a failed/non-converged MDS run),
        in which case only metadata is stored and ``confdist_row`` is -1."""
        missing = set(self._meta_columns) - set(meta)
        if missing:
            raise ValueError(f"meta is missing columns {sorted(missing)}")
        row_index = -1
        if confdist is not None:
            confdist = np.ascontiguousarray(confdist, dtype=_CONFDIST_DTYPE).ravel()
            if confdist.size != self.confdist_len:
                raise ValueError(
                    f"confdist length {confdist.size} != store confdist_len {self.confdist_len}"
                )
            if self._confdist_fh is None:
                self._confdist_fh = open(self.path / _CONFDIST_FILE, "ab")
            self._confdist_fh.write(confdist.tobytes())
            self._confdist_fh.flush()
            row_index = self._n_confdists
            self._n_confdists += 1
        with open(self.path / _META_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([meta[c] for c in self._meta_columns] + [row_index])

    def close(self) -> None:
        if self._confdist_fh is not None:
            self._confdist_fh.close()
            self._confdist_fh = None

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ reading
    def metadata(self) -> pd.DataFrame:
        """Return all metadata rows as a DataFrame (the lightweight table; no confdists)."""
        return pd.read_csv(self.path / _META_FILE)

    def _confdist_memmap(self) -> Optional[np.memmap]:
        if self._n_confdists == 0:
            return None
        return np.memmap(
            self.path / _CONFDIST_FILE, dtype=_CONFDIST_DTYPE, mode="r",
            shape=(self._n_confdists, self.confdist_len),
        )

    def confdist(self, row: int) -> np.ndarray:
        """Load a single confdist vector by its row index (a copy, not the memmap view)."""
        if row < 0 or row >= self._n_confdists:
            raise IndexError(f"confdist row {row} out of range [0, {self._n_confdists})")
        return np.array(self._confdist_memmap()[row])

    def iter_results(self) -> Iterator[Tuple[Dict[str, Any], Optional[np.ndarray]]]:
        """Lazily yield ``(meta_dict, confdist_or_None)`` for every stored record."""
        mm = self._confdist_memmap()
        meta = self.metadata()
        records = meta.to_dict("records")
        for rec in records:
            row = int(rec.pop(_ROW_COLUMN))
            yield rec, (np.array(mm[row]) if row >= 0 else None)

    def __len__(self) -> int:
        # number of metadata rows (not only the confdist-bearing ones)
        with open(self.path / _META_FILE, "r") as f:
            return max(sum(1 for _ in f) - 1, 0)  # minus header
