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
    # Pre-allocate: peek at the first image to determine flat size, then
    # fill row-by-row. tqdm covers all n paths so the bar reads n/n.
    n = len(paths)
    flat_size = load_image_rgb(paths[0]).size  # H*W*3
    X = np.empty((n, flat_size), dtype=np.float64)
    for i, p in enumerate(tqdm(paths, desc=f"sens_{variant}")):
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
