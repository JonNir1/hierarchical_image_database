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
from scipy.spatial.distance import pdist
from tqdm import tqdm

from analysis.rdms.common import image_paths, load_image_rgb, save_rdm


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
    pixels = []
    for p in tqdm(paths, desc=f"sens_{variant}"):
        img = load_image_rgb(p)          # (H, W, 3) uint8
        pixels.append(img.flatten().astype(np.float64))
    X = np.stack(pixels)                 # (725, H*W*3)

    print(f"[sensory] Pixel matrix {X.shape}. Computing pairwise Euclidean distances ...")
    condensed = pdist(X, metric="euclidean")

    short = "sens_pre" if variant == "pre_shine" else "sens_post"
    save_rdm(
        short,
        condensed,
        metric="euclidean",
        source="analysis.rdms.sensory",
        extra={"variant": variant, "pixel_matrix_shape": list(X.shape)},
    )
    print(f"[sensory] Saved D_{short}.npy  (length {len(condensed)})")
    return condensed


if __name__ == "__main__":
    build_sensory_rdm("pre_shine")
    build_sensory_rdm("post_shine")
