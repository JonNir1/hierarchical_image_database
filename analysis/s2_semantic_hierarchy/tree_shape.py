"""
The a priori hierarchy itself (0_tree_shape.ipynb is the front end): its shape, why KM distance is not
a function of how far up two images diverge, and the pooled bins every later notebook uses.

No RDM other than sem_km is involved: this notebook describes the reference, not agreement with it.

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import tree_shape as ts
    ts.pair_counts(c.tree_parts())
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analysis.s2_semantic_hierarchy.common import (
    LEVEL_NAMES, branch_strata, lca_bins, pooled_leaf_depths, tier_label,
)
from analysis.utils.hierarchy import Parts, lca_depth_condensed, path_depths

_TEMPLATE = "plotly_white"


def leaf_depth_counts(parts: Parts) -> pd.DataFrame:
    """Images per leaf depth, with the pooled tier each depth falls into."""
    depths = path_depths(parts).astype(int)
    counts = pd.Series(depths).value_counts().sort_index()
    return pd.DataFrame({
        "n_images": counts.to_numpy(),
        "pooled_tier": [tier_label(int(d)) for d in counts.index],
    }, index=pd.Index(counts.index, name="leaf_depth"))


def categories_per_level(parts: Parts) -> pd.DataFrame:
    """Per tree level: how many categories, and how many images each holds (min / median / max)."""
    rows = []
    for level, name in LEVEL_NAMES.items():
        sizes = Counter(p[:level] for p in parts if len(p) >= level)
        n = np.array(list(sizes.values()))
        rows.append({"level": level, "name": name, "n_categories": len(sizes),
                     "min_images": int(n.min()), "median_images": float(np.median(n)), "max_images": int(n.max())})
    return pd.DataFrame(rows).set_index("level")


def km_vs_lca(km: np.ndarray, parts: Parts, *, pooled: bool = False) -> pd.DataFrame:
    """Pair counts of KM distance (columns) per LCA depth (rows): several KM values per row = ragged tree."""
    lca = lca_bins(parts) if pooled else lca_depth_condensed(parts)
    index = pd.Series([tier_label(int(d)) for d in lca] if pooled else lca, name="lca_depth")
    return pd.crosstab(index, pd.Series(km.astype(int), name="km_distance"))


def pair_counts(parts: Parts) -> pd.DataFrame:
    """Pair counts per pooled LCA bin (rows) and branch-depth stratum (columns), with margins."""
    lca = lca_bins(parts)
    rows = pd.Series([tier_label(int(d)) for d in lca], name="lca_bin")
    table = pd.crosstab(rows, pd.Series(branch_strata(parts), name="stratum"), margins=True, margins_name="all")
    order = [tier_label(int(d)) for d in np.unique(lca)] + ["all"]
    return table.loc[order]


def deepest_tier_images(parts: Parts) -> pd.DataFrame:
    """Images at or below DEEP_TIER per top-two-level branch: where the pooled tier comes from."""
    deep = pooled_leaf_depths(parts) == pooled_leaf_depths(parts).max()
    counts = Counter("/".join(p[:3]) for p, d in zip(parts, deep) if d)
    return pd.Series(counts, name="n_images").sort_values(ascending=False).to_frame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_tree(parts: Parts) -> go.Figure:
    """Icicle of the directory tree, each node sized by its number of images."""
    counts: Counter = Counter()
    for p in parts:
        for depth in range(1, len(p) + 1):
            counts[p[:depth]] += 1
    nodes = sorted(counts, key=lambda n: (len(n), n))
    fig = go.Figure(go.Icicle(
        ids=["/".join(n) for n in nodes],
        labels=[n[-1] for n in nodes],
        parents=["/".join(n[:-1]) for n in nodes],
        values=[counts[n] for n in nodes],
        branchvalues="total",
        tiling={"orientation": "v"},
        hovertemplate="%{id}<br>%{value} images<extra></extra>",
        maxdepth=4,
    ))
    fig.update_layout(title="The curated hierarchy (node size = number of images; click to zoom)",
                      template=_TEMPLATE, height=620, margin={"t": 60, "l": 10, "r": 10, "b": 10})
    return fig


def fig_pair_counts(table: pd.DataFrame) -> go.Figure:
    """Heatmap of pair_counts() without its margins."""
    core = table.drop(index="all", columns="all")
    fig = go.Figure(go.Heatmap(
        z=core.to_numpy(), x=list(core.columns), y=list(core.index),
        colorscale="Greys", text=core.to_numpy(), texttemplate="%{text:,}", showscale=False,
    ))
    fig.update_layout(
        title="Image pairs per pooled LCA bin and branch-depth stratum",
        xaxis_title="branch-depth stratum (binned leaf depths of the two images)",
        yaxis_title="LCA depth (0 = diverge at the root)",
        template=_TEMPLATE, height=420, width=640,
    )
    return fig
