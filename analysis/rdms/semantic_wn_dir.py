"""
WordNet semantic RDM from manifest synset assignments.

Reads wn_synset_name from images/manifest.csv and computes all
N*(N-1)/2 pairwise WordNet shortest-path distances.

No neural-network classification step — synsets are derived directly
from each image's filename stem and parent directory, with manual
overrides for polysemous or missing cases.  See images/manifest.csv
column wn_synset_name and the assignment logic in
analysis/rdms/assign_wn_synsets.py (or the git history of manifest.csv
for the full provenance).

Supersedes semantic_wn.py (ResNet-50 + WordNet pipeline).

Output (to analysis/results/rdms/):
    D_sem_wn.npy

Usage (from repo root):
    python -m analysis.rdms.semantic_wn_dir
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from tqdm import tqdm

from analysis.rdms.common import load_manifest, save_rdm, _EXPECTED_N

_WN_FALLBACK_DIST = 30.0


@lru_cache(maxsize=None)
def _wn_distance(syn_a: "wn.Synset", syn_b: "wn.Synset") -> float:
    """WordNet shortest-path distance between two synsets (cached)."""
    d = syn_a.shortest_path_distance(syn_b)
    return float(d) if d is not None else _WN_FALLBACK_DIST


def build_wn_rdm() -> np.ndarray:
    """
    Build the WordNet semantic RDM from manifest synset assignments.

    Returns
    -------
    Condensed distance vector (float64, length 262_450)
    """
    manifest = load_manifest()

    if "wn_synset_name" not in manifest.columns:
        raise RuntimeError(
            "images/manifest.csv is missing the 'wn_synset_name' column. "
            "Run analysis/rdms/assign_wn_synsets.py to populate it."
        )

    synsets: list[wn.Synset | None] = []
    for name in manifest["wn_synset_name"]:
        if pd.isna(name):
            synsets.append(None)
        else:
            try:
                synsets.append(wn.synset(str(name)))
            except WordNetError:
                synsets.append(None)

    none_count = sum(s is None for s in synsets)
    if none_count:
        print(f"[semantic_wn_dir] {none_count} images had no resolvable synset.")

    n = len(synsets)
    if n != _EXPECTED_N:
        raise RuntimeError(f"Expected {_EXPECTED_N} images, got {n}.")

    print(f"[semantic_wn_dir] Computing pairwise WN distances for {n} images ...")
    k = n * (n - 1) // 2
    condensed = np.empty(k, dtype=np.float64)
    fallback_pairs = 0
    idx = 0
    for i in tqdm(range(n), desc="sem_wn_dir"):
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
        print(
            f"[semantic_wn_dir] {fallback_pairs} pairs used fallback "
            f"distance {_WN_FALLBACK_DIST}"
        )

    save_rdm(
        "sem_wn",
        condensed,
        metric="wordnet_shortest_path",
        source="analysis.rdms.semantic_wn_dir",
        extra={
            "synset_source": "images/manifest.csv:wn_synset_name",
            "wn_fallback_dist": _WN_FALLBACK_DIST,
        },
    )
    print(f"[semantic_wn_dir] Saved D_sem_wn.npy  (length {len(condensed)})")
    return condensed


if __name__ == "__main__":
    build_wn_rdm()
