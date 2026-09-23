"""
Hierarchical-clustering agreement between the perceptual SpAM geometry and the Kiani-Mur (KM)
directory tree (Q1, analysis 1b).

Ground truth is the tree itself, converted to a scipy linkage matrix by path_tree_to_linkage().
The perceptual side is an MDS embedding of the behavioral RDM (analysis.rdms.mds.run_mds_sweep,
which handles the missing pairs) that is then clustered two ways, reusing the machinery built for
the SpAM_Simulations reproducibility analysis rather than reimplementing it:

  agglomerative -- SpAM_Simulations.measures.cluster_stability.build_linkage over every linkage in
      DEFAULT_LINKAGES (prior simulation work found no decisive winner, so none is hardcoded).
      Whole-tree agreement is Baker's gamma of the two dendrograms' cophenetic ranks.
  HDBSCAN -- SpAM_Simulations.measures.density_clustering.hdbscan_labels on the same embedded
      distances. It yields flat labels, not a dendrogram, so agreement is scored per KM level
      (ARI / VI over the images HDBSCAN clustered) instead of Baker's gamma, and is descriptive.

Usage (from repo root):
    from analysis.rdms import hierarchical_clustering as hcl
    parts = hcl.path_parts()
    z_gt = hcl.path_tree_to_linkage(parts)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from analysis.rdms.hierarchy_comparison import path_parts
from analysis.rdms.mds import run_mds_sweep
from SpAM_Simulations.measures.cluster_stability import (
    DEFAULT_LINKAGES, baker_gamma, build_linkage, cophenetic_fidelity, cophenetic_ranks,
    variation_of_information,
)
from SpAM_Simulations.measures.density_clustering import (
    DEFAULT_MIN_CLUSTER_SIZES, NOISE_LABEL, clustered_ari, hdbscan_labels, noise_summary,
)

Parts = list[tuple[str, ...]]

__all__ = [
    "path_parts", "path_tree_to_linkage", "km_level_labels", "perceptual_embeddings",
    "agglomerative_agreement", "GammaShuffleResult", "dendrogram_shuffle_test",
    "hdbscan_level_agreement",
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


def perceptual_embeddings(
    condensed: np.ndarray, cohort: str, dims: Sequence[int], cache_dir: Path,
) -> dict[int, np.ndarray]:
    """Condensed Euclidean distances of the cached/fitted MDS configuration at each dimension.

    `cohort` ("pre"/"post") is the cache key run_mds_sweep uses in cohort_comparison, so existing
    fits are reused instead of refitted.
    """
    sweep = run_mds_sweep(condensed, list(dims), cohort, cache_dir)
    return {d: pdist(sweep[d]["conf"]) for d in dims}


def agglomerative_agreement(
    z_gt: np.ndarray, embedded: dict[int, np.ndarray], linkages: Sequence[str] = DEFAULT_LINKAGES,
) -> pd.DataFrame:
    """Baker's gamma against the GT tree, per (MDS dimension, linkage), with cophenetic fidelity."""
    gt_ranks = cophenetic_ranks(z_gt)
    rows = []
    for dim, dists in embedded.items():
        for method in linkages:
            z = build_linkage(dists, method)
            rows.append({
                "ndim": dim, "linkage": method,
                "baker_gamma": baker_gamma(gt_ranks, cophenetic_ranks(z)),
                "cophenetic_fidelity": cophenetic_fidelity(z, dists),
            })
    return pd.DataFrame(rows)


@dataclass
class GammaShuffleResult:
    observed: float
    null: np.ndarray
    p_value: float


def dendrogram_shuffle_test(
    perceptual_ranks: np.ndarray, parts: Parts, *, n_perm: int = 1000, seed: int = 0,
) -> GammaShuffleResult:
    """Baker's gamma of a fixed perceptual dendrogram against the GT tree, versus a null that
    permutes image identity in the GT tree (one-sided, +1 corrected). `perceptual_ranks` is
    cophenetic_ranks() of the perceptual linkage."""
    rng = np.random.default_rng(seed)
    observed = baker_gamma(cophenetic_ranks(path_tree_to_linkage(parts)), perceptual_ranks)
    null = np.empty(n_perm)
    for i in range(n_perm):
        permuted = [parts[j] for j in rng.permutation(len(parts))]
        null[i] = baker_gamma(cophenetic_ranks(path_tree_to_linkage(permuted)), perceptual_ranks)
    p_value = (float(np.sum(null >= observed)) + 1.0) / (n_perm + 1.0)
    return GammaShuffleResult(observed, null, p_value)


def hdbscan_level_agreement(
    embedded: dict[int, np.ndarray], parts: Parts, levels: Sequence[int],
    min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES,
) -> pd.DataFrame:
    """HDBSCAN flat labels vs KM category labels, per (ndim, min_cluster_size, KM level).

    ARI and VI are computed over the images that HDBSCAN clustered and that reach the KM level, so
    `frac_shared` must be read with them: an aggressive setting keeps only an easy core.
    """
    rows = []
    for dim, dists in embedded.items():
        for mcs in min_cluster_sizes:
            labels = hdbscan_labels(dists, mcs)
            noise = noise_summary(labels)
            for level in levels:
                km = km_level_labels(parts, level)
                agreement = clustered_ari(labels, km)
                shared = (labels != NOISE_LABEL) & (km != NOISE_LABEL)
                vi = (variation_of_information(labels[shared], km[shared])
                      if shared.sum() >= 2 else np.nan)
                rows.append({
                    "ndim": dim, "min_cluster_size": mcs, "km_level": level,
                    "n_clusters": noise["n_clusters"], "frac_noise": noise["frac_noise"],
                    "ari": agreement["ari_shared_clustered"], "vi_norm": vi,
                    "frac_shared": agreement["frac_shared_clustered"],
                })
    return pd.DataFrame(rows)
