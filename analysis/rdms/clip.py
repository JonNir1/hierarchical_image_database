"""
CLIP visual-semantic RDMs: pairwise cosine distance between ViT-B/32 output embeddings.

Uses OpenAI pretrained CLIP ViT-B/32 via open_clip.
Embeddings are taken from the image encoder output layer (following Shoham et al. 2024).

Outputs (to analysis/results/rdms/):
    D_clip_pre.npy   -- pre-SHINE images
    D_clip_post.npy  -- post-SHINE images

Requires:
    pip install open_clip_torch

Usage (from repo root):
    python -m analysis.rdms.clip
"""
from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image as PILImage
from scipy.spatial.distance import pdist
from tqdm import tqdm

from analysis.rdms.common import image_paths, load_image_rgb, save_rdm


def build_clip_rdm(variant: str) -> np.ndarray:
    """
    Build the CLIP visual-semantic RDM for the given SHINE variant.

    Parameters
    ----------
    variant : 'pre_shine' or 'post_shine'

    Returns
    -------
    Condensed cosine-distance vector (float64, length 262_450)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] Loading CLIP ViT-B/32 (openai) on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()

    paths = image_paths(variant)
    print(f"[clip] Encoding {len(paths)} images for variant '{variant}' ...")
    embeddings = []
    for p in tqdm(paths, desc=f"clip_{variant}"):
        img_pil = PILImage.fromarray(load_image_rgb(p))
        tensor = preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
        embeddings.append(emb.squeeze(0).cpu().float().numpy())

    E = np.stack(embeddings)   # (725, embedding_dim)
    print(f"[clip] Embedding matrix {E.shape}. Computing pairwise cosine distances ...")
    # pdist with 'cosine' = 1 - cos_similarity; handles normalisation internally
    condensed = pdist(E, metric="cosine")

    short = "clip_pre" if variant == "pre_shine" else "clip_post"
    save_rdm(
        short,
        condensed,
        metric="cosine",
        source="analysis.rdms.clip",
        extra={
            "variant": variant,
            "model": "ViT-B-32",
            "pretrained": "openai",
            "embedding_dim": E.shape[1],
        },
    )
    print(f"[clip] Saved D_{short}.npy  (length {len(condensed)})")
    return condensed


if __name__ == "__main__":
    build_clip_rdm("pre_shine")
    build_clip_rdm("post_shine")
