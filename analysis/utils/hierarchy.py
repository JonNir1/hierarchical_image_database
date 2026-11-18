"""
The directory-tree hierarchy encoded in the curated image paths.

images/<variant>/animate/animal/body/bird/duck3.png encodes domain -> mid-level -> basic level, so
the Kiani-Mur hierarchy is recoverable from the paths alone. These helpers take the manifest's
`curated_path` strings (Windows or POSIX separators) and return the directory components, their
depths, and the depth of every pair's lowest common ancestor (LCA).

Usage (from repo root):
    from analysis.utils.hierarchy import path_parts, lca_depth_condensed

    parts = path_parts(load_manifest()["curated_path"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.distance import squareform

Parts = list[tuple[str, ...]]


def path_parts(curated_paths: Iterable[str]) -> Parts:
    """Directory components of every path (filename dropped), in the order given."""
    return [tuple(Path(p.replace("\\", "/")).parts[:-1]) for p in curated_paths]


def path_depths(parts: Parts) -> np.ndarray:
    """Number of directory levels below the variant root, per image."""
    return np.array([len(p) for p in parts], dtype=np.float64)


def lca_depth_matrix(parts: Parts) -> np.ndarray:
    """
    (N, N) depth of each pair's lowest common ancestor directory.

    0 means the two images share no directory at all (they diverge at the root, i.e. animate vs
    inanimate); a larger value means they stay together deeper into the tree. The diagonal is
    zeroed, so read it only off-diagonal.
    """
    n = len(parts)
    lca = np.zeros((n, n), dtype=np.int16)
    for level in range(1, int(path_depths(parts).max()) + 1):
        groups: dict[tuple[str, ...], list[int]] = {}
        for i, p in enumerate(parts):
            if len(p) >= level:
                groups.setdefault(p[:level], []).append(i)
        for members in groups.values():
            if len(members) > 1:
                ix = np.array(members)
                lca[np.ix_(ix, ix)] = level
    np.fill_diagonal(lca, 0)
    return lca


def lca_depth_condensed(parts: Parts) -> np.ndarray:
    """Condensed (upper-triangle) form of lca_depth_matrix()."""
    return squareform(lca_depth_matrix(parts), checks=False)


def depth_is_ragged(parts: Parts) -> bool:
    """
    True if leaves sit at more than one depth.

    When the tree is ragged, KM distance is not a function of LCA depth alone: a pair in a
    finely-subdivided branch is pushed further apart than an equally related pair in a coarse one.
    """
    return len(np.unique(path_depths(parts))) > 1
