"""
Kiani-Mur semantic RDM: tree-edge distance via directory hierarchy LCA.

Distance between images i and j:
    d(i, j) = (depth_i - depth_lca) + (depth_j - depth_lca)

where depth is the number of directory segments in curated_path (excluding
the filename), and depth_lca is the depth of the lowest common ancestor
directory. Off-diagonal entries are floored at 1: two distinct images in the
same leaf folder are never distance 0.

Variant-agnostic (hierarchy is in the directory structure, not the pixels).

Output (to analysis/results/rdms/):
    D_sem_km.npy

Usage (from repo root):
    python -m analysis.rdms.semantic_km
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.distance import squareform
from tqdm import tqdm

from analysis.rdms.common import load_manifest, save_rdm


def _dir_parts(curated_path: str) -> tuple[str, ...]:
    """Directory-component parts of a manifest path (all segments except filename)."""
    p = Path(curated_path.replace("\\", "/"))
    return p.parts[:-1]


def _lca_depth(parts_a: tuple[str, ...], parts_b: tuple[str, ...]) -> int:
    """Number of matching prefix segments = depth of lowest common ancestor."""
    depth = 0
    for a, b in zip(parts_a, parts_b):
        if a == b:
            depth += 1
        else:
            break
    return depth


def build_km_rdm() -> np.ndarray:
    """
    Build the Kiani-Mur tree-edge-distance RDM.

    LCA matrix is built by iterating directory levels and grouping images
    that share a common prefix at each level. Per-level cost is O(g²) for
    a group of size g; worst case is O(n² × max_depth) but typical
    hierarchical data is much cheaper. Distance is then computed via
    np.add.outer (vectorised).

    Returns
    -------
    Condensed distance vector (float64, length 262_450)
    """
    df = load_manifest()
    dir_parts = [_dir_parts(p) for p in df["curated_path"]]
    depths = np.array([len(p) for p in dir_parts], dtype=np.float64)
    n = len(dir_parts)
    max_depth = int(depths.max())
    print(f"[semantic_km] Building LCA matrix for {n} images (max depth {max_depth}) ...")

    # Build LCA matrix in O(n * max_depth):
    # For each level l, images sharing the same path prefix of length l have
    # LCA >= l. Overwriting with increasing l leaves each cell at the deepest
    # common level — which is exactly the LCA depth.
    lca_matrix = np.zeros((n, n), dtype=np.float64)
    for level in tqdm(range(1, max_depth + 1), desc="lca levels"):
        groups: dict[tuple[str, ...], list[int]] = {}
        for i, parts in enumerate(dir_parts):
            if len(parts) >= level:
                key = parts[:level]
                groups.setdefault(key, []).append(i)
        for group_idx in groups.values():
            if len(group_idx) > 1:
                ix = np.array(group_idx)
                lca_matrix[np.ix_(ix, ix)] = level

    # Vectorised distance: d(i,j) = depth_i + depth_j - 2*lca(i,j)
    dist_matrix = np.add.outer(depths, depths) - 2.0 * lca_matrix
    # Floor off-diagonal at 1; keep diagonal at 0
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.maximum(dist_matrix, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)

    save_rdm(
        "sem_km",
        condensed,
        metric="km_tree_edge",
        source="analysis.rdms.semantic_km",
    )
    print(f"[semantic_km] Saved D_sem_km.npy  (length {len(condensed)})")
    return condensed


if __name__ == "__main__":
    build_km_rdm()
