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

    Returns
    -------
    Condensed distance vector (float64, length 262_450)
    """
    df = load_manifest()
    dir_parts = [_dir_parts(p) for p in df["curated_path"]]
    depths = [len(p) for p in dir_parts]
    n = len(dir_parts)
    print(f"[semantic_km] Computing tree-edge distances for {n} images ...")

    k = n * (n - 1) // 2
    condensed = np.empty(k, dtype=np.float64)
    idx = 0
    for i in tqdm(range(n), desc="sem_km"):
        for j in range(i + 1, n):
            lca = _lca_depth(dir_parts[i], dir_parts[j])
            d = (depths[i] - lca) + (depths[j] - lca)
            condensed[idx] = max(1, d)   # floor at 1 off-diagonal
            idx += 1

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
