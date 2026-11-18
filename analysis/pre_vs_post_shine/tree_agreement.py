"""
Pre- vs post-SHINE cohort comparison, tree-level agreement (analysis/pre_vs_post_shine/
4_semantic_hierarchy.ipynb and 5_ground_truth.ipynb): does the hierarchy implied by each cohort's
perceptual geometry match a reference hierarchy (the Kiani directory tree, or an external cohort's
dendrogram)?

Reference: path_tree_to_linkage() turns the directory tree itself into a scipy linkage matrix, exactly
rather than by clustering an RDM. Perceptual side: the cached MDS embedding of the behavioral RDM
(analysis.build_rdms.behavior.mds.run_mds_sweep, which handles the missing pairs), clustered
agglomeratively with average linkage. The v5 simulation report (SpAM_Simulations/sim_results/v5/report/
report_v5.src.html) finds average linkage outperforms ward and complete throughout (Baker's gamma
0.611 vs 0.579 and 0.471), so it is the only linkage used here. Whole-tree agreement is Baker's gamma of
the two dendrograms' cophenetic ranks, with an image-relabeling shuffle null. HDBSCAN yields flat
labels, not a dendrogram, so it is scored per KM level (ARI / VI over the images it clustered) and is
descriptive. The clustering and gamma machinery is reused from SpAM_Simulations.measures.

Usage (from repo root):
    from analysis.pre_vs_post_shine import tree_agreement as ta
    z_km = ta.path_tree_to_linkage(ta.manifest_parts())
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from analysis.build_rdms.behavior.mds import run_mds_sweep
from analysis.build_rdms.common import load_manifest
from analysis.pre_vs_post_shine.demographics_and_behavior import COHORTS, cohort_colors, cohort_label
from analysis.utils.disk_cache import cached_array, fingerprint
from analysis.utils.hierarchy import Parts, path_parts
from SpAM_Simulations.measures.cluster_stability import (
    baker_gamma, build_linkage, cophenetic_fidelity, cophenetic_ranks, variation_of_information,
)
from SpAM_Simulations.measures.density_clustering import (
    DEFAULT_MIN_CLUSTER_SIZES, NOISE_LABEL, clustered_ari, hdbscan_labels, noise_summary,
)

LINKAGE = "average"
MDS_DIMS = (3, 9, 15)

__all__ = [
    "LINKAGE", "MDS_DIMS", "manifest_parts", "path_tree_to_linkage", "km_level_labels", "perceptual_embeddings",
    "perceptual_tree", "agglomerative_agreement", "GammaShuffleResult", "dendrogram_shuffle_test",
    "tree_shuffle_test", "hdbscan_level_agreement", "fig_baker_gamma", "fig_hdbscan_agreement",
]


def manifest_parts() -> Parts:
    """Directory components of every manifest path, in manifest row order (the Kiani tree)."""
    return path_parts(load_manifest()["curated_path"])


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

    `cohort` ("pre"/"post") is the cache key notebook 3 uses with run_mds_sweep, so its existing
    fits are reused instead of refitted.
    """
    sweep = run_mds_sweep(condensed, list(dims), cohort, cache_dir)
    return {d: pdist(sweep[d]["conf"]) for d in dims}


def perceptual_tree(embedded_distances: np.ndarray) -> np.ndarray:
    """Average-linkage dendrogram of one MDS embedding's condensed distances."""
    return build_linkage(embedded_distances, LINKAGE)


def agglomerative_agreement(
    z_reference: np.ndarray, embedded: dict[int, np.ndarray], linkages: Sequence[str] = (LINKAGE,),
) -> pd.DataFrame:
    """Baker's gamma against the reference tree, per (MDS dimension, linkage), with cophenetic fidelity
    (how much of its own embedded geometry each dendrogram keeps, which qualifies the gamma)."""
    reference_ranks = cophenetic_ranks(z_reference)
    rows = []
    for dim, dists in embedded.items():
        for method in linkages:
            z = build_linkage(dists, method)
            rows.append({
                "ndim": dim, "linkage": method,
                "baker_gamma": baker_gamma(reference_ranks, cophenetic_ranks(z)),
                "cophenetic_fidelity": cophenetic_fidelity(z, dists),
            })
    return pd.DataFrame(rows)


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
                vi = variation_of_information(labels[shared], km[shared]) if shared.sum() >= 2 else np.nan
                rows.append({
                    "ndim": dim, "min_cluster_size": mcs, "km_level": level,
                    "n_clusters": noise["n_clusters"], "frac_noise": noise["frac_noise"],
                    "ari": agreement["ari_shared_clustered"], "vi_norm": vi,
                    "frac_shared": agreement["frac_shared_clustered"],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_TEMPLATE = "plotly_white"


def fig_baker_gamma(rows: pd.DataFrame, *, title: str) -> go.Figure:
    """Baker's gamma per MDS dimension and cohort (bars) against the 95th percentile of the shuffle
    null (grey ticks). `rows` has columns cohort, ndim, baker_gamma, null_p95."""
    fig = go.Figure()
    for cohort in COHORTS:
        data = rows[rows["cohort"] == cohort].sort_values("ndim")
        x = [f"d = {d}" for d in data["ndim"]]
        fig.add_trace(go.Bar(x=x, y=data["baker_gamma"], name=cohort_label(cohort),
                             marker_color=cohort_colors(cohort)["violin_line"]))
        fig.add_trace(go.Scatter(x=x, y=data["null_p95"], mode="markers", showlegend=False,
                                 marker={"symbol": "line-ew", "size": 22, "color": "grey", "line": {"width": 2}}))
    fig.update_layout(template=_TEMPLATE, barmode="group", height=420, yaxis={"title": "Baker's gamma", "range": [0, 1]},
                      title=title)
    return fig


def fig_hdbscan_agreement(table: pd.DataFrame, *, level_names: dict[int, str], ndim: int) -> go.Figure:
    """ARI over the images HDBSCAN clustered against KM categories, by min_cluster_size, one panel per KM level."""
    from plotly.subplots import make_subplots

    levels = list(level_names)
    fig = make_subplots(rows=1, cols=len(levels), shared_yaxes=True, subplot_titles=[level_names[v] for v in levels])
    data = table[table["ndim"] == ndim]
    for col, level in enumerate(levels, start=1):
        for cohort in COHORTS:
            d = data[(data["cohort"] == cohort) & (data["km_level"] == level)]
            fig.add_trace(go.Scatter(x=d["min_cluster_size"], y=d["ari"], mode="lines+markers", name=cohort_label(cohort),
                                     line_color=cohort_colors(cohort)["violin_line"], showlegend=col == 1), row=1, col=col)
    fig.update_xaxes(type="log", title_text="min_cluster_size")
    fig.update_yaxes(title_text="ARI over clustered images", row=1, col=1)
    fig.update_layout(template=_TEMPLATE, height=420)
    return fig
