"""
WordNet semantic RDM: pairwise WordNet hypernym shortest-path distance.

Pipeline:
  1. Classify each pre-SHINE image with ImageNet-pretrained ResNet-50 (top-1).
     Results are cached in analysis/rdms/imagenet_classifications.csv (tracked
     in git) so the forward pass runs only once.
  2. Map ImageNet class index -> WordNet synset via imagenet_class_index.json
     (downloaded automatically to analysis/results/ on first run).
  3. Compute pairwise WordNet shortest_path_distance (cached per synset pair).

Variant-agnostic: classification runs on pre-SHINE images only; the resulting
RDM is used for both cohorts.

Output (to analysis/results/rdms/):
    D_sem_wn.npy

Requires:
    pip install torch torchvision nltk
    python -m nltk.downloader wordnet omw-1.4

Usage (from repo root):
    python -m analysis.rdms.semantic_wn
"""
from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image as PILImage
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from analysis.rdms.common import RESULTS_DIR, image_paths, load_image_rgb, load_manifest, save_rdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Cache lives next to this module so it can be tracked in git
CLASSIFICATIONS_PATH = Path(__file__).parent / "imagenet_classifications.csv"

# imagenet_class_index.json maps str(idx) -> [wnid, class_name]; downloaded once
_CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/"
    "imagenet_class_index.json"
)
_CLASS_INDEX_CACHE = Path(__file__).parent / "imagenet_class_index.json"

# Fallback distance when no WordNet path exists between two synsets
_WN_FALLBACK_DIST = 30.0


# ---------------------------------------------------------------------------
# ImageNet class index
# ---------------------------------------------------------------------------


def _load_class_index() -> dict[str, list[str]]:
    """Load imagenet_class_index.json, downloading to analysis/rdms/ if absent."""
    if not _CLASS_INDEX_CACHE.exists():
        _CLASS_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[semantic_wn] Downloading imagenet_class_index.json ...")
        urllib.request.urlretrieve(_CLASS_INDEX_URL, _CLASS_INDEX_CACHE)
    with open(_CLASS_INDEX_CACHE) as f:
        return json.load(f)


def _wnid_to_synset(wnid: str):
    """Convert an ImageNet WordNet ID (e.g. 'n02119789') to an NLTK Synset."""
    pos = wnid[0]          # always 'n' for ImageNet
    offset = int(wnid[1:])
    return wn.synset_from_pos_and_offset(pos, offset)


@lru_cache(maxsize=None)
def _wn_distance(syn_a, syn_b) -> float:
    """
    WordNet shortest-path distance between two synsets (cached).
    Returns _WN_FALLBACK_DIST if no path exists.
    """
    d = syn_a.shortest_path_distance(syn_b)
    return float(d) if d is not None else _WN_FALLBACK_DIST


# ---------------------------------------------------------------------------
# ResNet-50 classifier
# ---------------------------------------------------------------------------


def _classify_images(paths: list[Path]) -> list[int]:
    """Classify each image with ResNet-50; return list of top-1 class indices."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    transform = weights.transforms()
    device = torch.device("cpu")   # 725 images is fast enough on CPU

    indices: list[int] = []
    for p in tqdm(paths, desc="ResNet50"):
        img_pil = PILImage.fromarray(load_image_rgb(p))
        tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        indices.append(int(logits.argmax(dim=1).item()))
    return indices


# ---------------------------------------------------------------------------
# Classification cache
# ---------------------------------------------------------------------------


def _build_classifications_cache() -> pd.DataFrame:
    """
    Run ResNet-50 on pre-SHINE images, resolve WordNet synsets, and save to
    CLASSIFICATIONS_PATH (analysis/rdms/imagenet_classifications.csv).

    Returns the resulting DataFrame.
    """
    paths = image_paths("pre_shine")
    print(f"[semantic_wn] Classifying {len(paths)} pre-SHINE images with ResNet-50 ...")
    class_indices = _classify_images(paths)

    class_index = _load_class_index()
    manifest = load_manifest()[["curated_path"]]

    records = []
    for idx in tqdm(class_indices, desc="idx->synset"):
        wnid, class_name = class_index[str(idx)]
        try:
            syn = _wnid_to_synset(wnid)
            synset_name = syn.name()
        except Exception:
            synset_name = None
        records.append({
            "imagenet_class_idx":  idx,
            "imagenet_wnid":       wnid,
            "imagenet_class_name": class_name,
            "wordnet_synset_name": synset_name,
        })

    df = pd.concat([manifest.reset_index(drop=True),
                    pd.DataFrame(records)], axis=1)
    df.to_csv(CLASSIFICATIONS_PATH, index=False)
    print(f"[semantic_wn] Saved classification cache -> {CLASSIFICATIONS_PATH}")
    return df


def load_or_build_classifications() -> pd.DataFrame:
    """
    Return the ImageNet classification DataFrame, building and caching it if
    analysis/rdms/imagenet_classifications.csv does not yet exist.

    Columns: curated_path, imagenet_class_idx, imagenet_wnid,
             imagenet_class_name, wordnet_synset_name
    """
    if CLASSIFICATIONS_PATH.exists():
        print(f"[semantic_wn] Loading classification cache from {CLASSIFICATIONS_PATH}")
        return pd.read_csv(CLASSIFICATIONS_PATH)
    return _build_classifications_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_wn_rdm() -> np.ndarray:
    """
    Build the WordNet semantic RDM.

    Returns
    -------
    Condensed distance vector (float64, length 262_450)
    """
    clf = load_or_build_classifications()

    # Reconstruct NLTK Synset objects from cached synset names
    synsets = []
    none_count = 0
    for name in clf["wordnet_synset_name"]:
        if pd.isna(name):
            synsets.append(None)
            none_count += 1
        else:
            try:
                synsets.append(wn.synset(str(name)))
            except Exception:
                synsets.append(None)
                none_count += 1
    if none_count:
        print(f"[semantic_wn] {none_count} images had no resolvable synset.")

    n = len(synsets)
    print(f"[semantic_wn] Computing pairwise WordNet distances for {n} images ...")
    k = n * (n - 1) // 2
    condensed = np.empty(k, dtype=np.float64)
    fallback_pairs = 0
    idx = 0
    for i in tqdm(range(n), desc="sem_wn"):
        for j in range(i + 1, n):
            sa, sb = synsets[i], synsets[j]
            if sa is None or sb is None:
                condensed[idx] = _WN_FALLBACK_DIST
                fallback_pairs += 1
            elif sa == sb:
                condensed[idx] = 0.0
            else:
                condensed[idx] = _wn_distance(sa, sb)
            idx += 1

    if fallback_pairs:
        print(f"[semantic_wn] {fallback_pairs} pairs used fallback distance {_WN_FALLBACK_DIST}")

    save_rdm(
        "sem_wn",
        condensed,
        metric="wordnet_shortest_path",
        source="analysis.rdms.semantic_wn",
        extra={
            "classifier": "resnet50_IMAGENET1K_V2",
            "variant_classified": "pre_shine",
            "classifications_cache": str(CLASSIFICATIONS_PATH),
            "wn_fallback_dist": _WN_FALLBACK_DIST,
        },
    )
    print(f"[semantic_wn] Saved D_sem_wn.npy  (length {len(condensed)})")
    return condensed


if __name__ == "__main__":
    build_wn_rdm()
