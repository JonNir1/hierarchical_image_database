"""
Small disk cache for expensive, deterministic results (permutation nulls, bootstraps): a result is
stored with a fingerprint of the inputs that produced it and reused only when the fingerprint
matches, so changing the subjects, n or seed recomputes instead of returning stale numbers.

`cached_array` keeps the .npz layout analysis/s1_pre_vs_post_shine/geometry.py always wrote (the array
under `key`, plus `fingerprint`), so caches written before this module existed stay valid.

Usage (from repo root):
    from analysis.utils.disk_cache import cached_array, cached_frame, fingerprint

    null = cached_array(path, fingerprint(ids, n_perm, seed), lambda: draw_null(...))
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def fingerprint(*parts: object) -> str:
    """SHA-1 over `parts`: arrays by their bytes, sequences by their joined items, anything else by repr."""
    digest = hashlib.sha1()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part).tobytes())
        elif isinstance(part, (list, tuple)):
            digest.update("|".join(map(str, part)).encode())
        else:
            digest.update(repr(part).encode())
        digest.update(b"#")
    return digest.hexdigest()


def cached_array(
    path: Path | None, fingerprint: str, compute: Callable[[], np.ndarray], *, key: str = "null",
) -> np.ndarray:
    """The array stored at `path` if its fingerprint matches, else `compute()` stored there and returned.
    `path=None` disables caching."""
    if path is None:
        return compute()
    path = Path(path)
    if path.exists():
        with np.load(path) as cached:
            if str(cached["fingerprint"]) == fingerprint:
                return cached[key]
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: value, "fingerprint": fingerprint})
    return value


def cached_frame(path: Path | None, fingerprint: str, compute: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """DataFrame analogue of cached_array(): pickled with its fingerprint (a local, trusted cache)."""
    if path is None:
        return compute()
    path = Path(path)
    if path.exists():
        with open(path, "rb") as handle:
            stored = pickle.load(handle)
        if stored["fingerprint"] == fingerprint:
            return stored["frame"]
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump({"fingerprint": fingerprint, "frame": value}, handle)
    return value
