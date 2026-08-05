"""
CLIP visual-semantic RDMs: pairwise cosine distance between ViT-B/32 output embeddings.

Uses OpenAI pretrained CLIP ViT-B/32 via open_clip.
Embeddings are taken from the image encoder output layer (following Shoham et al. 2024).

Model config: 'ViT-B-32-quickgelu', not 'ViT-B-32'. OpenAI trained CLIP with
QuickGELU activations; open_clip later split the two into separate configs,
leaving plain 'ViT-B-32' on standard GELU. Pairing 'ViT-B-32' with
pretrained='openai' therefore loads OpenAI's weights into a network whose
activation function differs from the one they were trained under, which
open_clip flags at load time as a QuickGELU mismatch. Measured across the full
725-image set, that mismatch shifted the RDM by mean |delta d| = 0.023 against a
mean distance of 0.36, giving Spearman rho = 0.943 (pre) and 0.938 (post)
against the correctly-configured RDM: a small but systematic distortion, not
noise. '-quickgelu' is the config faithful to the pre-registered
"OpenAI pretrained CLIP ViT-B/32".

Outputs (to analysis/results/rdms/):
    E_clip_pre.npy   -- pre-SHINE image embeddings  (725, embedding_dim)
    E_clip_post.npy  -- post-SHINE image embeddings (725, embedding_dim)
    D_clip_pre.npy   -- pre-SHINE pairwise cosine-distance RDM
    D_clip_post.npy  -- post-SHINE pairwise cosine-distance RDM

Requires:
    pip install open_clip_torch

Usage (from repo root):
    python -m analysis.rdms.clip
"""
from __future__ import annotations

import numpy as np
import open_clip
import torch
from tqdm import tqdm

from analysis.rdms.common import cosine_distances, image_paths, open_as_rgb_pil, save_embeddings

# See the module docstring: '-quickgelu' matches the activation OpenAI's
# weights were trained with. Do not drop the suffix.
_MODEL_NAME = "ViT-B-32-quickgelu"
_PRETRAINED = "openai"


def build_clip_rdm(variant: str) -> np.ndarray:
    """
    Build the CLIP visual-semantic RDM for the given SHINE variant.

    Encodes all images once, persists the raw embedding matrix, then derives
    the pairwise cosine-distance RDM from those embeddings.

    Parameters
    ----------
    variant : 'pre_shine' or 'post_shine'

    Returns
    -------
    Condensed cosine-distance vector (float64, length 262_450)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] Loading CLIP {_MODEL_NAME} ({_PRETRAINED}) on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED, device=device
    )
    model.eval()

    paths = image_paths(variant)
    print(f"[clip] Encoding {len(paths)} images for variant '{variant}' ...")
    embeddings = []
    for p in tqdm(paths, desc=f"clip_{variant}"):
        tensor = preprocess(open_as_rgb_pil(p)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
        embeddings.append(emb.squeeze(0).cpu().float().numpy())

    E = np.stack(embeddings)   # (725, embedding_dim)
    short = "clip_pre" if variant == "pre_shine" else "clip_post"
    meta = {"variant": variant, "model": _MODEL_NAME, "pretrained": _PRETRAINED}

    print(f"[clip] Embedding matrix {E.shape}. Saving embeddings ...")
    save_embeddings(short, E, source="analysis.rdms.clip", extra=meta)
    print(f"[clip] Saved E_{short}.npy")

    print("[clip] Computing pairwise cosine distances ...")
    condensed = cosine_distances(
        E,
        save_result=True,
        name=short,
        source="analysis.rdms.clip",
        extra={**meta, "embedding_dim": E.shape[1]},
    )
    print(f"[clip] Saved D_{short}.npy  (length {len(condensed)})")
    return condensed


if __name__ == "__main__":
    build_clip_rdm("pre_shine")
    build_clip_rdm("post_shine")
