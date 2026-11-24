"""
Generic dendrogram tools shared by the s1 and s2 analyses: the directory tree as an exact scipy linkage
matrix, per-level category labels, the average-linkage tree of an embedding's distances, Baker's
gamma between two dendrograms with an image-relabeling shuffle null, and HDBSCAN's flat partitions
scored against the tree's categories at each level.

Average linkage is the only linkage used: the v5 simulation report (SpAM_Simulations/sim_results/v5/
report/report_v5.src.html) finds it outperforms ward and complete throughout (Baker's gamma 0.611 vs
0.579 and 0.471). HDBSCAN complements it as in that report: it yields flat labels rather than a dendrogram,
and may leave images unassigned (noise), so it is scored per tree level, over the images it clustered,
and reported with its noise fraction (descriptive). The clustering and gamma machinery is reused from
SpAM_Simulations.measures.

Usage (from repo root):
    from analysis.utils import dendrogram as dg
    z_km = dg.path_tree_to_linkage(parts)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from analysis.utils.disk_cache import cached_array, fingerprint
from analysis.utils.hierarchy import Parts
from SpAM_Simulations.measures.cluster_stability import (
    baker_gamma, build_linkage, cophenetic_ranks, variation_of_information,
)
from SpAM_Simulations.measures.density_clustering import (
    DEFAULT_MIN_CLUSTER_SIZES, NOISE_LABEL, clustered_ari, hdbscan_labels, noise_summary,
)

LINKAGE = "average"

__all__ = [
    "LINKAGE", "path_tree_to_linkage", "km_level_labels", "perceptual_tree", "GammaShuffleResult",
    "dendrogram_shuffle_test", "tree_shuffle_test", "hdbscan_level_agreement",
]


def path_tree_to_linkage(parts: Parts) -> np.ndarray:
    """Binary scipy linkage matrix encoding the directory tree given by `parts`.

    Every directory becomes a merge point at height (max_depth + 1 - depth), so a pair's
    cophenetic distance falls as the pair shares a deeper directory. A directory with several
    children (images sitting directly in it plus subdirectories) is resolved into a chain of
    binary merges at that one height, ordered by index / name for reproducibility: the chain order
    never changes any pair's cophenetic distance.
    """
    n = len(parts)
    top = max(len(p) for p in parts) + 1
    leaves: dict[tuple[str, ...], list[int]] = {}
    subdirs: dict[tuple[str, ...], set[str]] = {}
    for i, p in enumerate(parts):
        leaves.setdefault(p, []).append(i)
        for depth in range(len(p)):
            subdirs.setdefault(p[:depth], set()).add(p[depth])

    sizes: dict[int, int] = {i: 1 for i in range(n)}
    merges: list[tuple[int, int, float, int]] = []

    def merge(a: int, b: int, height: float) -> int:
        node = n + len(merges)
        sizes[node] = sizes[a] + sizes[b]
        merges.append((a, b, height, sizes[node]))
        return node

    def build(prefix: tuple[str, ...]) -> int:
        members = list(leaves.get(prefix, []))
        members += [build(prefix + (name,)) for name in sorted(subdirs.get(prefix, ()))]
        height = float(top - len(prefix))
        acc = members[0]
        for member in members[1:]:
            acc = merge(acc, member, height)
        return acc

    build(())
    order = sorted(range(len(merges)), key=lambda k: (merges[k][2], k))
    new_id = {n + k: n + rank for rank, k in enumerate(order)}
    remap = lambda node: new_id.get(node, node)
    return np.array(
        [[remap(merges[k][0]), remap(merges[k][1]), merges[k][2], merges[k][3]] for k in order],
        dtype=np.float64,
    )

def km_level_labels(parts: Parts, level: int) -> np.ndarray:
    """Integer KM category per image at directory `level`; -1 for images shallower than `level`."""
    codes: dict[tuple[str, ...], int] = {}
    labels = np.full(len(parts), NOISE_LABEL, dtype=np.int64)
    for i, p in enumerate(parts):
        if len(p) >= level:
            labels[i] = codes.setdefault(p[:level], len(codes))
    return labels

def perceptual_tree(embedded_distances: np.ndarray) -> np.ndarray:
    """Average-linkage dendrogram of one MDS embedding's condensed distances."""
    return build_linkage(embedded_distances, LINKAGE)

@dataclass
class GammaShuffleResult:
    observed: float
    null: np.ndarray
    p_value: float

    def summary(self) -> dict[str, float]:
        return {"baker_gamma": self.observed, "null_mean": float(self.null.mean()), "null_sd": float(self.null.std()),
                "null_p95": float(np.quantile(self.null, 0.95)), "p_value": self.p_value}


def _shuffle_result(observed: float, null: np.ndarray) -> GammaShuffleResult:
    return GammaShuffleResult(observed, null, (float(np.sum(null >= observed)) + 1.0) / (len(null) + 1.0))


def dendrogram_shuffle_test(
    perceptual_ranks: np.ndarray, parts: Parts, *, n_perm: int = 1000, seed: int = 0,
    cache_path: Path | None = None,
) -> GammaShuffleResult:
    """Baker's gamma of a fixed perceptual dendrogram against the directory tree, versus a null that
    permutes image identity in the directory tree (one-sided, +1 corrected). `perceptual_ranks` is
    cophenetic_ranks() of the perceptual linkage."""
    def draw_null() -> np.ndarray:
        rng = np.random.default_rng(seed)
        draws = np.empty(n_perm)
        for i in range(n_perm):
            permuted = [parts[j] for j in rng.permutation(len(parts))]
            draws[i] = baker_gamma(cophenetic_ranks(path_tree_to_linkage(permuted)), perceptual_ranks)
        return draws

    observed = baker_gamma(cophenetic_ranks(path_tree_to_linkage(parts)), perceptual_ranks)
    null = cached_array(cache_path, fingerprint(perceptual_ranks, parts, n_perm, seed), draw_null)
    return _shuffle_result(observed, null)


def tree_shuffle_test(
    z_fixed: np.ndarray, z_shuffled: np.ndarray, *, n_perm: int = 1000, seed: int = 0,
    cache_path: Path | None = None,
) -> GammaShuffleResult:
    """Baker's gamma between two dendrograms over the same images, versus a null that permutes image
    identity in `z_shuffled` (one-sided, +1 corrected): the form for any second tree, e.g. an
    external cohort's."""
    fixed_ranks = cophenetic_ranks(z_fixed)

    def draw_null() -> np.ndarray:
        rng = np.random.default_rng(seed)
        coph = squareform(cophenet(z_shuffled))
        draws = np.empty(n_perm)
        for i in range(n_perm):
            p = rng.permutation(coph.shape[0])
            draws[i] = baker_gamma(fixed_ranks, rankdata(squareform(coph[np.ix_(p, p)], checks=False)))
        return draws

    observed = baker_gamma(fixed_ranks, cophenetic_ranks(z_shuffled))
    null = cached_array(cache_path, fingerprint(z_fixed, z_shuffled, n_perm, seed), draw_null)
    return _shuffle_result(observed, null)


def hdbscan_level_agreement(
    rdms: dict, parts: Parts, levels: Sequence[int],
    min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES, *, key: str = "rdm",
) -> pd.DataFrame:
    """HDBSCAN flat labels of each complete condensed RDM vs the tree's category labels, one row per
    (RDM, min_cluster_size, tree level); the RDM's dict key goes in column `key`.

    ARI and VI are computed over the images that HDBSCAN clustered and that reach the level, so
    `frac_shared` must be read with them: an aggressive setting keeps only an easy core.
    """
    rows = []
    for name, dists in rdms.items():
        for mcs in min_cluster_sizes:
            labels = hdbscan_labels(dists, mcs)
            noise = noise_summary(labels)
            for level in levels:
                km = km_level_labels(parts, level)
                agreement = clustered_ari(labels, km)
                shared = (labels != NOISE_LABEL) & (km != NOISE_LABEL)
                vi = variation_of_information(labels[shared], km[shared]) if shared.sum() >= 2 else np.nan
                rows.append({
                    key: name, "min_cluster_size": mcs, "km_level": level,
                    "n_clusters": noise["n_clusters"], "frac_noise": noise["frac_noise"],
                    "ari": agreement["ari_shared_clustered"], "vi_norm": vi,
                    "frac_shared": agreement["frac_shared_clustered"],
                })
    return pd.DataFrame(rows)
