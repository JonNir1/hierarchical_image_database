"""
WordNet semantic RDM: pairwise WordNet hypernym shortest-path distance.

Pipeline:
  1. Classify each pre-SHINE image with ImageNet-pretrained ResNet-50 (top-3).
     For each image the top-3 predictions are compared against the WordNet
     concept implied by the image's filename stem, parent directory, and their
     compound (e.g. ball/tennis.png -> "tennis ball").  The prediction whose
     synset is closest in the WordNet hierarchy is selected; this corrects
     systematic ResNet errors such as butterfly images being classified as
     hair slides.
     Results are cached in analysis/rdms/imagenet_classifications.csv (tracked
     in git) so the forward pass runs only once.
  2. Map ImageNet class index -> WordNet synset via imagenet_class_index.json
     (downloaded automatically to analysis/rdms/ on first run).
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
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet50, ResNet50_Weights
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from tqdm import tqdm

from analysis.rdms.common import image_paths, open_as_rgb_pil, load_manifest, save_rdm

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


@lru_cache(maxsize=None)
def _wn_distance(syn_a, syn_b) -> float:
    """
    WordNet shortest-path distance between two synsets (cached).
    Returns _WN_FALLBACK_DIST if no path exists.
    """
    d = syn_a.shortest_path_distance(syn_b)
    return float(d) if d is not None else _WN_FALLBACK_DIST


# ---------------------------------------------------------------------------
# Path-concept helpers (used for top-k selection)
# ---------------------------------------------------------------------------

def _path_tokens(curated_path: str) -> tuple[str, str]:
    """Return (filename_stem, parent_dirname) for a manifest curated_path."""
    p = Path(curated_path.replace("\\", "/"))
    stem   = re.sub(r'\d+$', '', p.stem).lower()               # e.g. "tennis"
    parent = p.parts[-2].lower() if len(p.parts) >= 2 else ""  # e.g. "ball"
    return stem, parent


def _candidate_synsets(curated_path: str) -> set:
    """
    All noun synsets for the filename stem, parent directory, and their two
    compound orderings (e.g. "tennis ball" and "ball tennis").
    """
    stem, parent = _path_tokens(curated_path)
    syns: set = set()
    for w in [stem, parent, f"{parent} {stem}", f"{stem} {parent}"]:
        w = w.strip()
        if w:
            syns.update(wn.synsets(w, pos=wn.NOUN))
            syns.update(wn.synsets(w.replace(" ", "_"), pos=wn.NOUN))
    return syns


def _dist_to_concept(curated_path: str, syn: "wn.Synset") -> float | None:
    """
    Minimum WordNet path distance from any candidate concept synset
    (stem / dirname / compound) to the given classified synset.
    Returns None if no candidate concept has a WordNet noun synset.
    """
    cands = _candidate_synsets(curated_path)
    if not cands:
        return None
    dists = [d for c in cands if (d := c.shortest_path_distance(syn)) is not None]
    return min(dists) if dists else None


# ---------------------------------------------------------------------------
# ResNet-50 classifier
# ---------------------------------------------------------------------------


def _classify_images(paths: list[Path], k: int = 3) -> list[list[int]]:
    """Classify each image with ResNet-50; return top-k class indices per image."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    transform = weights.transforms()
    device = torch.device("cpu")   # 725 images is fast enough on CPU

    all_indices: list[list[int]] = []
    for p in tqdm(paths, desc=f"ResNet50 top-{k}"):
        tensor = transform(open_as_rgb_pil(p)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        topk = logits.topk(k, dim=1).indices.squeeze(0).tolist()
        all_indices.append(topk)
    return all_indices


# ---------------------------------------------------------------------------
# Classification cache
# ---------------------------------------------------------------------------


def _build_classifications_cache() -> pd.DataFrame:
    """
    Run ResNet-50 top-3 on pre-SHINE images, select the prediction whose
    WordNet synset is closest to the filename/dirname concept, and save to
    CLASSIFICATIONS_PATH (analysis/rdms/imagenet_classifications.csv).

    The 'selected_rank' column records which top-k prediction was chosen
    (1 = top-1 was best, 2 or 3 = a lower-ranked prediction was closer to
    the image's filename/directory concept).

    Returns the resulting DataFrame.
    """
    paths = image_paths("pre_shine")
    print(f"[semantic_wn] Classifying {len(paths)} pre-SHINE images with ResNet-50 top-3 ...")
    topk_indices = _classify_images(paths, k=3)

    class_index = _load_class_index()
    manifest    = load_manifest()[["curated_path"]]

    records = []
    for curated_path, topk in tqdm(
        zip(manifest["curated_path"], topk_indices), total=len(manifest),
        desc="selecting best prediction"
    ):
        best_idx       = topk[0]   # fallback: top-1
        best_rank      = 1
        best_dist      = float("inf")
        best_syn_name: str | None = None

        for rank, idx in enumerate(topk, 1):
            wnid, _ = class_index[str(idx)]
            try:
                syn  = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
                dist = _dist_to_concept(curated_path, syn)
                if dist is not None and dist < best_dist:
                    best_dist, best_idx, best_rank, best_syn_name = dist, idx, rank, syn.name()
            except WordNetError:
                pass

        # If no candidate concept had a WN synset, resolve the chosen top-k synset directly
        if best_syn_name is None:
            wnid, _ = class_index[str(best_idx)]
            try:
                best_syn_name = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:])).name()
            except WordNetError:
                pass

        wnid_best, class_name_best = class_index[str(best_idx)]
        records.append({
            "imagenet_class_idx":  best_idx,
            "imagenet_wnid":       wnid_best,
            "imagenet_class_name": class_name_best,
            "wordnet_synset_name": best_syn_name,
            "selected_rank":       best_rank,
        })

    df = pd.concat([manifest.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    df.to_csv(CLASSIFICATIONS_PATH, index=False)
    print(f"[semantic_wn] Saved classification cache -> {CLASSIFICATIONS_PATH}")
    return df


def load_or_build_classifications() -> pd.DataFrame:
    """
    Return the ImageNet classification DataFrame, building and caching it if
    analysis/rdms/imagenet_classifications.csv does not yet exist.

    On load, validates row count, curated_path order, and schema against the
    current manifest; raises RuntimeError if anything is inconsistent.

    Columns: curated_path, imagenet_class_idx, imagenet_wnid,
             imagenet_class_name, wordnet_synset_name, selected_rank
    """
    if CLASSIFICATIONS_PATH.exists():
        print(f"[semantic_wn] Loading classification cache from {CLASSIFICATIONS_PATH}")
        clf = pd.read_csv(CLASSIFICATIONS_PATH)
        manifest = load_manifest()
        if len(clf) != len(manifest):
            raise RuntimeError(
                f"Classification cache has {len(clf)} rows, expected {len(manifest)}. "
                f"Delete {CLASSIFICATIONS_PATH} and re-run to rebuild."
            )
        if clf["curated_path"].tolist() != manifest["curated_path"].tolist():
            raise RuntimeError(
                f"Classification cache image order does not match current manifest. "
                f"Delete {CLASSIFICATIONS_PATH} and re-run to rebuild."
            )
        if "selected_rank" not in clf.columns:
            raise RuntimeError(
                f"Classification cache is stale (missing 'selected_rank' column from "
                f"top-3 re-classification). Delete {CLASSIFICATIONS_PATH} and re-run."
            )
        return clf
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
    for name in clf["wordnet_synset_name"]:
        if pd.isna(name):
            synsets.append(None)
        else:
            try:
                synsets.append(wn.synset(str(name)))
            except WordNetError:
                synsets.append(None)

    none_count = sum(s is None for s in synsets)
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
