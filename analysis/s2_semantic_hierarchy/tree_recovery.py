"""
Does the hierarchy implied by each RDM's own geometry reproduce the a priori tree?
(6_tree_recovery.ipynb is the front end.)

Notebooks 2 to 5 compare the RDM entries with the tree. This one asks the complementary question of
structure: cluster each RDM agglomeratively (average linkage, the only linkage used, see
analysis.utils.dendrogram) and compare the dendrogram it yields with the exact tree
(dendrogram.path_tree_to_linkage of the directory paths).

Complete RDMs only. Hierarchical clustering needs every pair, so the raw joint behavioral RDM (~13% of
pairs unobserved) cannot be clustered and is set aside; its SMACOF embedding (behav_joint_mds, complete)
stands for behavior. The embeddings at fixed dimensionality used in s1 (d = 3, 9, 15) are added for
comparison when their fits are already cached (never refitted here).

Whole-tree agreement is Baker's gamma (Pearson correlation of the two dendrograms' cophenetic ranks,
so it is k-free and dominated by the coarse structure). Its null permutes image identity in the exact
tree, which keeps the tree's shape and destroys the image-to-position assignment. One set of
permutations is shared by all RDMs (a paired null) and is applied to the tree's cophenetic rank matrix
directly (ranks are invariant under relabeling), which is equivalent to dendrogram_shuffle_test but
much faster for many RDMs. Cophenetic fidelity (Spearman between an RDM and its own dendrogram's
cophenetic distances) says how tree-like each RDM is, which qualifies the gamma. Flat cuts compare each
dendrogram cut into as many clusters as the tree has categories at a level with the tree's category
labels (adjusted Rand index). Because average linkage peels off tiny outlier clusters, a second,
outlier-robust cut is reported alongside: the dendrogram is cut at the smallest k for which the number of
clusters holding at least MIN_CLUSTER_SIZE images reaches the level's category count, and ARI is taken over
the images in those clusters only (descriptive; it asks whether the substantial clusters are the tree's).

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import tree_recovery as tr
    trees = tr.rdm_trees(rdms)
    tr.gamma_table(trees, c.tree_parts(), n_perm=1000)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from analysis.s2_semantic_hierarchy import common as c
from analysis.utils import plotting
from analysis.utils.dendrogram import km_level_labels, path_tree_to_linkage, perceptual_tree
from analysis.utils.disk_cache import cached_array, fingerprint
from analysis.utils.hierarchy import Parts
from SpAM_Simulations.measures.cluster_stability import baker_gamma, cophenetic_fidelity, cophenetic_ranks
from SpAM_Simulations.measures.density_clustering import NOISE_LABEL

EMBEDDING_DIMS = (3, 9, 15)
MIN_CLUSTER_SIZE = 5
MAX_CUT = 300
_TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# The RDMs to cluster
# ---------------------------------------------------------------------------


def split_complete(rdms: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    """(RDMs with no NaN, names of the RDMs left out because some pairs are unobserved)."""
    kept = {n: r for n, r in rdms.items() if not np.isnan(r).any()}
    return kept, [n for n in rdms if n not in kept]


def embedding_name(ndim: int) -> str:
    return f"behav_joint_mds{ndim}"


def cached_embeddings(behav: np.ndarray, dims: tuple[int, ...] = EMBEDDING_DIMS) -> dict[str, np.ndarray]:
    """Embedded distances of the joint behavioral RDM at each of `dims`; {} unless all their SMACOF fits
    are already cached (nothing is fitted here)."""
    from analysis.s2_semantic_hierarchy import mds_dimensionality as md

    key = f"behav_joint_{fingerprint(behav)[:10]}"
    if not all((md.MDS_CACHE_DIR / f"mds_{key}_d{d}.npz").exists() for d in dims):
        return {}
    fits = md.sweep(behav, dims)
    return {embedding_name(d): fits[d]["confdist"] for d in dims}


def rdm_trees(rdms: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Average-linkage dendrogram of each (complete) RDM."""
    return {n: perceptual_tree(r) for n, r in rdms.items()}


# ---------------------------------------------------------------------------
# Whole-tree agreement
# ---------------------------------------------------------------------------


def _null_gammas(tree_ranks: np.ndarray, rdm_ranks: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """(n_perm, n_rdms) Baker's gamma of each RDM's cophenetic ranks (rows of `rdm_ranks`) with the
    tree's, after permuting image identity in the tree; one permutation per draw, shared by the RDMs."""
    rank_matrix = squareform(tree_ranks)
    mean, sd = tree_ranks.mean(), tree_ranks.std()
    centered = rdm_ranks - rdm_ranks.mean(axis=1, keepdims=True)
    unit = centered / (rdm_ranks.std(axis=1, keepdims=True) * rdm_ranks.shape[1])
    rng = np.random.default_rng(seed)
    draws = np.empty((n_perm, len(rdm_ranks)))
    for i in range(n_perm):
        p = rng.permutation(rank_matrix.shape[0])
        permuted = squareform(rank_matrix[np.ix_(p, p)], checks=False)
        draws[i] = unit @ ((permuted - mean) / sd)
    return draws


def gamma_table(
    trees: dict[str, np.ndarray], parts: Parts, *, n_perm: int = 1000, seed: int = 0, cache_path: Path | None = None,
) -> pd.DataFrame:
    """Baker's gamma of each RDM's dendrogram against the exact tree, with the image-shuffle null's mean
    and 2.5 / 97.5 / 95th percentiles and the one-sided +1-corrected p value."""
    tree_ranks = cophenetic_ranks(path_tree_to_linkage(parts))
    rdm_ranks = np.vstack([cophenetic_ranks(z) for z in trees.values()])
    null = cached_array(
        cache_path, fingerprint(list(trees), rdm_ranks, parts, n_perm, seed),
        lambda: _null_gammas(tree_ranks, rdm_ranks, n_perm, seed),
    )
    rows = []
    for k, name in enumerate(trees):
        observed = baker_gamma(tree_ranks, rdm_ranks[k])
        draws = null[:, k]
        rows.append({
            "rdm": name, "baker_gamma": observed, "null_mean": float(draws.mean()),
            "null_lo": float(np.quantile(draws, 0.025)), "null_hi": float(np.quantile(draws, 0.975)),
            "null_p95": float(np.quantile(draws, 0.95)),
            "p_value": (float(np.sum(draws >= observed)) + 1.0) / (len(draws) + 1.0),
        })
    return pd.DataFrame(rows).set_index("rdm")


def cophenetic_table(trees: dict[str, np.ndarray], rdms: dict[str, np.ndarray]) -> pd.Series:
    """Spearman correlation of each RDM with the cophenetic distances of its own dendrogram."""
    return pd.Series({n: cophenetic_fidelity(z, rdms[n]) for n, z in trees.items()}, name="cophenetic_fidelity")


# ---------------------------------------------------------------------------
# Flat cuts
# ---------------------------------------------------------------------------


def level_category_counts(parts: Parts, levels: list[int] | None = None) -> dict[int, int]:
    """Number of tree categories at each level."""
    levels = list(c.LEVEL_NAMES) if levels is None else levels
    return {level: int(km_level_labels(parts, level).max()) + 1 for level in levels}


def flat_cut(z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Flat clustering of dendrogram `z` into (at most, under tied heights) `n_clusters` clusters."""
    return fcluster(z, t=n_clusters, criterion="maxclust")


def robust_cut(z: np.ndarray, n_categories: int) -> tuple[np.ndarray, int]:
    """(cluster labels, k) of the cut at the smallest k whose count of clusters with at least
    MIN_CLUSTER_SIZE images reaches `n_categories` (k = MAX_CUT if never reached)."""
    for k in range(n_categories, MAX_CUT + 1):
        cut = flat_cut(z, k)
        if (np.bincount(cut) >= MIN_CLUSTER_SIZE).sum() >= n_categories:
            return cut, k
    return cut, MAX_CUT


def flat_cut_ari(trees: dict[str, np.ndarray], parts: Parts, levels: list[int] | None = None) -> pd.DataFrame:
    """Per (RDM, level): ARI between the dendrogram cut at the level's category count and the tree's
    category labels (over the images that reach the level), the number of clusters the cut yielded and the
    share of images in its largest cluster (a cut of average linkage often peels off tiny clusters)."""
    rows = []
    for level, k in level_category_counts(parts, levels).items():
        truth = km_level_labels(parts, level)
        reach = truth != NOISE_LABEL
        for name, z in trees.items():
            cut = flat_cut(z, k)
            robust, k_used = robust_cut(z, k)
            big = np.isin(robust, np.flatnonzero(np.bincount(robust) >= MIN_CLUSTER_SIZE)) & reach
            rows.append({
                "rdm": name, "level": level, "n_categories": k, "n_clusters": int(len(np.unique(cut))),
                "robust_k": k_used, "robust_frac_kept": float(big.sum() / len(big)),
                "robust_ari": float(adjusted_rand_score(truth[big], robust[big])),
                "largest_cluster": float(np.bincount(cut).max() / len(cut)),
                "ari": float(adjusted_rand_score(truth[reach], cut[reach])),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_gamma(table: pd.DataFrame) -> go.Figure:
    """Baker's gamma per RDM (marker) over the image-shuffle null's 95% range (grey bar)."""
    names = plotting.rdm_legend_order(list(table.index))
    fig = go.Figure()
    for k, name in enumerate(names):
        x = plotting.rdm_label(name)
        fig.add_trace(go.Scatter(
            x=[x, x], y=[table.loc[name, "null_lo"], table.loc[name, "null_hi"]], mode="lines",
            line={"color": "lightgrey", "width": 10}, name="image-shuffle null, 95% range", showlegend=k == 0,
        ))
    for name in names:
        kwargs = plotting.rdm_scatter_kwargs(name, size=13)
        kwargs.pop("line")
        fig.add_trace(go.Scatter(x=[plotting.rdm_label(name)], y=[table.loc[name, "baker_gamma"]], mode="markers",
                                 showlegend=False, **kwargs))
    fig.update_layout(template=_TEMPLATE, height=440, yaxis_title="Baker's gamma with the exact tree",
                      xaxis={"categoryorder": "array", "categoryarray": [plotting.rdm_label(n) for n in names]},
                      legend={"orientation": "h", "y": 1.1})
    return fig


def fig_ari_heatmap(table: pd.DataFrame) -> go.Figure:
    """ARI of each RDM's flat cut (rows) per tree level (columns), values printed in the cells."""
    names = plotting.rdm_legend_order(list(table["rdm"].unique()))
    wide = table.pivot(index="rdm", columns="level", values="ari").loc[names]
    counts = table.groupby("level")["n_categories"].first()
    fig = go.Figure(go.Heatmap(
        z=wide.to_numpy(), x=[f"{c.LEVEL_NAMES[lv]}<br>({counts[lv]} clusters)" for lv in wide.columns],
        y=[plotting.rdm_tick_label(n) for n in names], zmin=0, zmax=1, colorscale="Viridis",
        text=np.round(wide.to_numpy(), 2), texttemplate="%{text}", colorbar={"title": "ARI"},
    ))
    fig.update_layout(template=_TEMPLATE, height=140 + 45 * len(names), yaxis={"autorange": "reversed"})
    return fig
