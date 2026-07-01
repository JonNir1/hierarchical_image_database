"""
Sensory RDMs: pairwise Euclidean distance between flattened RGB pixel vectors.

No luminance renormalisation is applied; SHINE handles that.

Outputs (to analysis/results/rdms/):
    D_sens_pre.npy   -- pre-SHINE images
    D_sens_post.npy  -- post-SHINE images

Usage (from repo root):
    python -m analysis.rdms.sensory
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from analysis.rdms.common import euclidean_distances, image_paths, load_image_rgb


def build_sensory_rdm(variant: str) -> np.ndarray:
    """
    Build the sensory (pixel-Euclidean) RDM for the given SHINE variant.

    Parameters
    ----------
    variant : 'pre_shine' or 'post_shine'

    Returns
    -------
    Condensed distance vector (float64, length 262_450)
    """
    paths = image_paths(variant)
    print(f"[sensory] Loading {len(paths)} images for variant '{variant}' ...")
    # Pre-allocate: load the first image to determine flat size, then fill the
    # remaining rows. tqdm starts at 1 (row 0 already loaded) so the bar still
    # reads n/n without re-reading paths[0] from disk.
    n = len(paths)
    first = load_image_rgb(paths[0]).flatten()
    X = np.empty((n, first.size), dtype=np.float64)
    X[0] = first
    for i, p in enumerate(tqdm(paths[1:], desc=f"sens_{variant}", initial=1, total=n), start=1):
        X[i] = load_image_rgb(p).flatten()

    print(f"[sensory] Pixel matrix {X.shape}. Computing pairwise Euclidean distances ...")
    short = "sens_pre" if variant == "pre_shine" else "sens_post"
    condensed = euclidean_distances(
        X,
        save_result=True,
        name=short,
        source="analysis.rdms.sensory",
        extra={"variant": variant, "pixel_matrix_shape": list(X.shape)},
    )
    print(f"[sensory] Saved D_{short}.npy  (length {len(condensed)})")
    return condensed


if __name__ == "__main__":
    build_sensory_rdm("pre_shine")
    build_sensory_rdm("post_shine")
