"""Compact, streamable on-disk storage for MDS-sweep results.

The pre-refactor notebook appended every result (including a full ``confdist`` vector of
``N(N-1)/2`` float64 values) to a single growing ``mds_results.pkl`` via repeated
``pickle.dump``. For the full sweep that file reached multiple GB and had to be read back
in its entirety. ``ResultStore`` replaces that with a small directory:

* ``store_info.json`` - the fixed confdist length and dtype (+ the conf geometry, if enabled).
* ``meta.csv``        - one human-readable row per result (parameters + niter/stress/status
                        + the row index of its confdist, or -1 if none).
* ``confdists.f32``   - a flat little-endian float32 binary; row ``i`` is one confdist vector.
* ``confs.f32``       - OPTIONAL, same layout; row ``i`` is one MDS *configuration* (the fitted
                        coordinates), zero-padded to ``n_images * max_ndim``.

Records are appended incrementally (crash-safe, flushed per write) and read back lazily
(both binaries are memory-mapped, so individual vectors are paged in on demand instead of
loading everything into RAM). Storing confdists as float32 also halves them versus the
original float64 pickle.

**Configurations (``confs.f32``).** A sweep fits several target dimensionalities, so a
configuration's true width (``n_images x ndim``) varies from record to record while
``ResultStore`` needs fixed-width rows. Rows are therefore sized for ``max_ndim`` and
zero-padded; ``ndim`` is already a metadata column, so :meth:`ResultStore.conf` trims each row
back to ``(n_images, ndim)`` on read. Coordinates are far smaller than distances (for 725
images and ``max_ndim=10``: 7,250 floats versus 262,450), so this costs a few percent of the
store's size and enables configuration-space comparisons (e.g. Procrustes between two cohorts)
that a distance vector cannot support.

**Format versions.** Version 1 stores have no ``confs.f32`` and no conf keys in
``store_info.json``; :meth:`ResultStore.open` reads them unchanged, and :meth:`conf` raises a
clear error rather than returning nonsense.

**Conf-only stores.** ``confdists.f32`` may be absent entirely, and such a store opens and serves
:meth:`conf` normally. This is a supported read mode, not a corrupt store: ``confdist`` is exactly
``pdist(conf)``, so for any analysis that starts from coordinates the large file is redundant. At
725 images and ``max_ndim=20`` a conf row is 58 KB against 1.05 MB for a confdist row, so a 480-fit
sweep is ~28 MB rather than ~500 MB - which is what makes the cluster analysis practical to run
locally on a downloaded store. The record count therefore comes from ``meta.csv`` rather than from
the binary's size, and :meth:`confdist` raises a directed error when the file is missing.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INFO_FILE = "store_info.json"
_META_FILE = "meta.csv"
_CONFDIST_FILE = "confdists.f32"
_CONF_FILE = "confs.f32"
_CONFDIST_DTYPE = np.float32
_ROW_COLUMN = "confdist_row"
_FORMAT_VERSION = 2


class ResultStore:
    """Append-only, lazily-read store for sweep results keyed by arbitrary metadata.

    ``n_images``/``max_ndim`` are optional: supply both to also store each fit's MDS
    configuration in ``confs.f32`` (see the module docstring). Omit them for a
    confdist-only store, which is byte-identical to the format-version-1 layout.
    """

    def __init__(self, path: str | Path, confdist_len: int, meta_columns: List[str],
                 n_images: Optional[int] = None, max_ndim: Optional[int] = None):
        self.path = Path(path)
        self.confdist_len = int(confdist_len)
        self._meta_columns = list(meta_columns)
        self._n_confdists = 0
        self._confdist_fh = None  # lazily opened append handle
        self.n_images = int(n_images) if n_images is not None else None
        self.max_ndim = int(max_ndim) if max_ndim is not None else None
        self._conf_fh = None

    @property
    def stores_conf(self) -> bool:
        """True if this store also records MDS configurations (``confs.f32``)."""
        return self.n_images is not None and self.max_ndim is not None

    @property
    def conf_len(self) -> int:
        """Padded row length of one configuration, or 0 if configurations aren't stored."""
        return self.n_images * self.max_ndim if self.stores_conf else 0

    @property
    def has_confdists(self) -> bool:
        """True if ``confdists.f32`` is present and non-empty.

        False for a **conf-only download**: since ``confdist == pdist(conf)`` the large file is
        redundant for any analysis that starts from coordinates, so it is routinely left behind.
        `conf()` works regardless; `confdist()` does not.
        """
        f = self.path / _CONFDIST_FILE
        return f.exists() and f.stat().st_size > 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, path: str | Path, confdist_len: int, meta_columns: List[str],
               overwrite: bool = False, n_images: Optional[int] = None,
               max_ndim: Optional[int] = None) -> "ResultStore":
        """Create a new store directory. ``meta_columns`` are the per-record fields
        (e.g. the experiment parameters plus 'rep', 'ndim'); status fields may also appear.

        Pass both ``n_images`` and ``max_ndim`` to additionally store each fit's MDS
        configuration; omitting them yields a confdist-only store.
        """
        path = Path(path)
        info_path = path / _INFO_FILE
        if info_path.exists() and not overwrite:
            raise FileExistsError(f"A store already exists at {path}; pass overwrite=True or use open().")
        if (n_images is None) != (max_ndim is None):
            raise ValueError("`n_images` and `max_ndim` must be given together (or both omitted)")
        if n_images is not None and (n_images < 1 or max_ndim < 1):
            raise ValueError(f"`n_images`/`max_ndim` must be positive, got {n_images}/{max_ndim}")
        path.mkdir(parents=True, exist_ok=True)
        if _ROW_COLUMN in meta_columns:
            raise ValueError(f"'{_ROW_COLUMN}' is reserved and must not be a metadata column")
        info = {
            "format_version": _FORMAT_VERSION,
            "confdist_len": int(confdist_len),
            "confdist_dtype": np.dtype(_CONFDIST_DTYPE).str,
            "meta_columns": list(meta_columns),
        }
        if n_images is not None:
            info["n_images"] = int(n_images)
            info["max_ndim"] = int(max_ndim)
            info["conf_len"] = int(n_images) * int(max_ndim)
        info_path.write_text(json.dumps(info, indent=2))
        # (re)initialise data files
        (path / _CONFDIST_FILE).write_bytes(b"")
        if n_images is not None:
            (path / _CONF_FILE).write_bytes(b"")
        with open(path / _META_FILE, "w", newline="") as f:
            csv.writer(f).writerow(list(meta_columns) + [_ROW_COLUMN])
        return cls(path, confdist_len, meta_columns, n_images, max_ndim)

    @classmethod
    def open(cls, path: str | Path) -> "ResultStore":
        """Open an existing store for reading and/or further appends.

        Format-version-1 stores (no ``n_images``/``max_ndim`` in ``store_info.json``, no
        ``confs.f32``) open unchanged as confdist-only stores.
        """
        path = Path(path)
        info = json.loads((path / _INFO_FILE).read_text())
        store = cls(path, info["confdist_len"], info["meta_columns"],
                    info.get("n_images"), info.get("max_ndim"))

        # Record count comes from meta.csv, not from the confdists file size, so that a store
        # downloaded WITHOUT `confdists.f32` still opens. That download is the normal case for the
        # cluster analysis: `confdist == pdist(conf)`, and a conf row is ~20x smaller (n_images *
        # max_ndim vs n_images^2/2 floats), so pulling confs alone is ~28 MB against ~500 MB for a
        # 480-fit sweep. meta.csv is authoritative for row indices in any case.
        meta = pd.read_csv(path / _META_FILE)
        rows = meta[_ROW_COLUMN] if _ROW_COLUMN in meta.columns else pd.Series(dtype="int64")
        rows = rows[rows >= 0]
        store._n_confdists = int(rows.max()) + 1 if len(rows) else 0

        if store.has_confdists:
            itemsize = np.dtype(_CONFDIST_DTYPE).itemsize
            on_disk = (path / _CONFDIST_FILE).stat().st_size // (store.confdist_len * itemsize)
            if on_disk != store._n_confdists:
                logger.warning(
                    "%s holds %d records but meta.csv indexes %d; using meta.csv. A truncated or "
                    "partially-synced confdists.f32 will raise on read.",
                    _CONFDIST_FILE, on_disk, store._n_confdists,
                )
        return store

    # ------------------------------------------------------------------ writing
    def append(self, meta: Dict[str, Any], confdist: Optional[np.ndarray] = None,
               conf: Optional[np.ndarray] = None) -> None:
        """Append one result. ``confdist`` may be None (e.g. a failed/non-converged MDS run),
        in which case only metadata is stored and ``confdist_row`` is -1.

        On a conf-storing store, ``conf`` (the fitted ``(n_images, ndim)`` coordinates) is
        **required** whenever ``confdist`` is given: the two binaries share one row index, so
        allowing one without the other would silently desynchronise them. It is zero-padded to
        ``max_ndim`` columns before being written.
        """
        missing = set(self._meta_columns) - set(meta)
        if missing:
            raise ValueError(f"meta is missing columns {sorted(missing)}")
        if conf is not None and not self.stores_conf:
            raise ValueError("this store was created without `n_images`/`max_ndim`, so it cannot store `conf`")
        if self.stores_conf and confdist is not None and conf is None:
            raise ValueError("`conf` is required alongside `confdist` on a conf-storing store")
        row_index = -1
        if confdist is not None:
            confdist = np.ascontiguousarray(confdist, dtype=_CONFDIST_DTYPE).ravel()
            if confdist.size != self.confdist_len:
                raise ValueError(
                    f"confdist length {confdist.size} != store confdist_len {self.confdist_len}"
                )
            padded = self._pad_conf(conf) if self.stores_conf else None
            if self._confdist_fh is None:
                self._confdist_fh = open(self.path / _CONFDIST_FILE, "ab")
            self._confdist_fh.write(confdist.tobytes())
            self._confdist_fh.flush()
            if padded is not None:
                if self._conf_fh is None:
                    self._conf_fh = open(self.path / _CONF_FILE, "ab")
                self._conf_fh.write(padded.tobytes())
                self._conf_fh.flush()
            row_index = self._n_confdists
            self._n_confdists += 1
        with open(self.path / _META_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([meta[c] for c in self._meta_columns] + [row_index])

    def _pad_conf(self, conf: np.ndarray) -> np.ndarray:
        """Validate a ``(n_images, ndim)`` configuration and zero-pad it to ``max_ndim`` columns."""
        conf = np.ascontiguousarray(conf, dtype=_CONFDIST_DTYPE)
        if conf.ndim != 2:
            raise ValueError(f"conf must be a 2-D (n_images, ndim) array, got shape {conf.shape}")
        rows, ndim = conf.shape
        if rows != self.n_images:
            raise ValueError(f"conf has {rows} rows != store n_images {self.n_images}")
        if not (1 <= ndim <= self.max_ndim):
            raise ValueError(f"conf ndim {ndim} must be in [1, max_ndim={self.max_ndim}]")
        if ndim < self.max_ndim:
            conf = np.pad(conf, ((0, 0), (0, self.max_ndim - ndim)))
        return np.ascontiguousarray(conf).ravel()

    def close(self) -> None:
        for attr in ("_confdist_fh", "_conf_fh"):
            fh = getattr(self, attr)
            if fh is not None:
                fh.close()
                setattr(self, attr, None)

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ reading
    def metadata(self) -> pd.DataFrame:
        """Return all metadata rows as a DataFrame (the lightweight table; no confdists)."""
        return pd.read_csv(self.path / _META_FILE)

    def _confdist_memmap(self) -> Optional[np.memmap]:
        if self._n_confdists == 0 or not self.has_confdists:
            return None
        return np.memmap(
            self.path / _CONFDIST_FILE, dtype=_CONFDIST_DTYPE, mode="r",
            shape=(self._n_confdists, self.confdist_len),
        )

    def confdist(self, row: int) -> np.ndarray:
        """Load a single confdist vector by its row index (a copy, not the memmap view)."""
        if not self.has_confdists:
            raise ValueError(
                f"{_CONFDIST_FILE} is missing from {self.path}, so reconstructed distances cannot "
                f"be read. This is expected for a conf-only download; use `conf(row, ndim)` and "
                f"`scipy.spatial.distance.pdist` instead, which is equivalent."
            )
        if row < 0 or row >= self._n_confdists:
            raise IndexError(f"confdist row {row} out of range [0, {self._n_confdists})")
        return np.array(self._confdist_memmap()[row])

    def _conf_memmap(self) -> Optional[np.memmap]:
        if self._n_confdists == 0 or not self.stores_conf:
            return None
        return np.memmap(
            self.path / _CONF_FILE, dtype=_CONFDIST_DTYPE, mode="r",
            shape=(self._n_confdists, self.conf_len),
        )

    def conf(self, row: int, ndim: Optional[int] = None) -> np.ndarray:
        """Load a single MDS configuration by its row index, as ``(n_images, ndim)``.

        Rows are stored zero-padded to ``max_ndim`` columns (see the module docstring), so pass
        the record's ``ndim`` (from ``meta.csv``) to trim the padding off. Without it the full
        padded ``(n_images, max_ndim)`` array is returned.
        """
        if not self.stores_conf:
            raise ValueError(
                "this store does not contain configurations (created without `n_images`/`max_ndim`; "
                "format-version-1 stores never have them)"
            )
        if row < 0 or row >= self._n_confdists:
            raise IndexError(f"conf row {row} out of range [0, {self._n_confdists})")
        flat = np.array(self._conf_memmap()[row]).reshape(self.n_images, self.max_ndim)
        if ndim is None:
            return flat
        if not (1 <= ndim <= self.max_ndim):
            raise ValueError(f"ndim {ndim} must be in [1, max_ndim={self.max_ndim}]")
        return flat[:, :ndim]

    def iter_results(self) -> Iterator[Tuple[Dict[str, Any], Optional[np.ndarray]]]:
        """Lazily yield ``(meta_dict, confdist_or_None)`` for every stored record.

        The confdist is ``None`` both for metadata-only records (a failed fit) and for every record
        of a conf-only store, since there is no file to read it from.
        """
        mm = self._confdist_memmap()
        meta = self.metadata()
        records = meta.to_dict("records")
        for rec in records:
            row = int(rec.pop(_ROW_COLUMN))
            yield rec, (np.array(mm[row]) if row >= 0 and mm is not None else None)

    def __len__(self) -> int:
        # number of metadata rows (not only the confdist-bearing ones)
        with open(self.path / _META_FILE, "r") as f:
            return max(sum(1 for _ in f) - 1, 0)  # minus header
