"""
Pre- vs post-SHINE cohort comparison, tree-level agreement (analysis/s1_pre_vs_post_shine/
4_semantic_hierarchy.ipynb and 5_ground_truth.ipynb): does the hierarchy implied by each cohort's
perceptual geometry match a reference hierarchy (the Kiani directory tree, or an external cohort's
dendrogram)?

Reference: path_tree_to_linkage() turns the directory tree itself into a scipy linkage matrix, exactly
rather than by clustering an RDM. Perceptual side: the cached MDS embedding of the behavioral RDM
(analysis.utils.mds.run_mds_sweep, which handles the missing pairs), clustered
agglomeratively with average linkage. The v5 simulation report (SpAM_Simulations/sim_results/v5/report/
report_v5.src.html) finds average linkage outperforms ward and complete throughout (Baker's gamma
0.611 vs 0.579 and 0.471), so it is the only linkage used here. Whole-tree agreement is Baker's gamma of
the two dendrograms' cophenetic ranks, with an image-relabeling shuffle null. HDBSCAN yields flat
labels, not a dendrogram, so it is scored per KM level (ARI / VI over the images it clustered) and is
descriptive. The clustering and gamma machinery is reused from SpAM_Simulations.measures.

Usage (from repo root):
    from analysis.s1_pre_vs_post_shine import tree_agreement as ta
    z_km = ta.path_tree_to_linkage(ta.manifest_parts())
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist

from analysis.utils import dendrogram
from analysis.utils.dendrogram import (
    LINKAGE, GammaShuffleResult, dendrogram_shuffle_test, km_level_labels, path_tree_to_linkage, perceptual_tree,
    tree_shuffle_test,
)
from analysis.utils.mds import run_mds_sweep
from analysis.s0_build_rdms.common import manifest_parts
from analysis.s1_pre_vs_post_shine.demographics_and_behavior import COHORTS, cohort_colors, cohort_label, cohort_symbol
from analysis.utils.hierarchy import Parts
from SpAM_Simulations.measures.cluster_stability import (
    baker_gamma, build_linkage, cophenetic_fidelity, cophenetic_ranks,
)
from SpAM_Simulations.measures.density_clustering import DEFAULT_MIN_CLUSTER_SIZES

MDS_DIMS = (3, 9, 15)

__all__ = [
    "LINKAGE", "MDS_DIMS", "manifest_parts", "path_tree_to_linkage", "km_level_labels", "perceptual_embeddings",
    "perceptual_tree", "agglomerative_agreement", "GammaShuffleResult", "dendrogram_shuffle_test",
    "tree_shuffle_test", "hdbscan_level_agreement", "fig_baker_gamma", "fig_hdbscan_agreement",
]


def perceptual_embeddings(
    condensed: np.ndarray, cohort: str, dims: Sequence[int], cache_dir: Path,
) -> dict[int, np.ndarray]:
    """Condensed Euclidean distances of the cached/fitted MDS configuration at each dimension.

    `cohort` ("pre"/"post") is the cache key notebook 3 uses with run_mds_sweep, so its existing
    fits are reused instead of refitted.
    """
    sweep = run_mds_sweep(condensed, list(dims), cohort, cache_dir)
    return {d: pdist(sweep[d]["conf"]) for d in dims}


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


def hdbscan_level_agreement(
    embedded: dict[int, np.ndarray], parts: Parts, levels: Sequence[int],
    min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES,
) -> pd.DataFrame:
    """HDBSCAN flat labels vs KM category labels, per (ndim, min_cluster_size, KM level): the shared
    analysis.utils.dendrogram.hdbscan_level_agreement keyed by embedding dimensionality."""
    return dendrogram.hdbscan_level_agreement(embedded, parts, levels, min_cluster_sizes, key="ndim")


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
                                     line_color=cohort_colors(cohort)["violin_line"],
                                     marker_symbol=cohort_symbol(cohort), showlegend=col == 1), row=1, col=col)
    fig.update_xaxes(type="log", title_text="min_cluster_size")
    fig.update_yaxes(title_text="ARI over clustered images", row=1, col=1)
    fig.update_layout(template=_TEMPLATE, height=420)
    return fig
