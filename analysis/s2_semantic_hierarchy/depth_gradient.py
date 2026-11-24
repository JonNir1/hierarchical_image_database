"""
Does distance grow as two images diverge higher up the tree? (3_depth_gradient.ipynb is the front end.)

Every RDM is summarised by its mean distance per pooled LCA bin ({0, 1, 2, 3, >=4}, 0 = the pair diverges
at the root), divided by its own value at bin 0 so that pixel Euclidean, cosine and SpAM distances share
one axis. A curve that falls from left to right is an RDM that places hierarchically closer images closer.

Method choices:
  * NaN-aware. Unobserved pairs (behav_joint is ~13% NaN) are ignored: each bin mean is a nanmean over
    the observed pairs of that bin.
  * Uncertainty is an image bootstrap (stimuli are the sampling unit): the condensed LCA-bin vector (and
    the branch-depth stratum code) is resampled together with the RDM, so every draw recomputes the bins
    of the resampled pairs. Bands and CIs are 95% percentiles.
  * The rank slope condenses the gradient into one number: minus the Spearman rho between a pair's
    distance and its LCA bin, over its observed pairs. LCA bin grows as pairs share deeper ancestors, so
    a distance that falls with depth has a negative rho; it is reported sign-flipped, so positive means
    "distance grows as pairs diverge higher up the tree". sem_km is not 1 because the tree is ragged.
  * Strata. Within a branch-depth stratum (3x3, 3x>=4, >=4x>=4) some LCA bins are empty (3x3 has no LCA-0
    pairs), so each stratum's curve is normalised to its own first non-empty bin. Strata are confounded
    with domain (every animate image is deep; every depth-3 image is inanimate), see 0_tree_shape.

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import common as c, depth_gradient as dg
    rdms = c.load_rdm_set(c.RAW_SET)
    dg.rank_slope_table(rdms, n_boot=1000, cache_dir=c.cache_path("depth_gradient"))
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.s2_semantic_hierarchy import common
from analysis.utils import rdm_style
from analysis.utils.disk_cache import fingerprint
from analysis.utils.hierarchy import Parts
from analysis.utils.stats import image_bootstrap, percentile_ci, spearman_rho

_TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------


def bin_means(x: np.ndarray, bin_index: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean of `x` per bin over its observed (non-NaN) pairs; NaN for a bin with none."""
    observed = ~np.isnan(x)
    sums = np.bincount(bin_index[observed], weights=x[observed], minlength=n_bins)
    counts = np.bincount(bin_index[observed], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def normalised_profile(x: np.ndarray, bin_index: np.ndarray, n_bins: int, ref: int = 0) -> np.ndarray:
    """bin_means() divided by the mean of bin `ref`."""
    means = bin_means(x, bin_index, n_bins)
    return means / means[ref]


def rank_slope(x: np.ndarray, bin_index: np.ndarray) -> float:
    """Minus Spearman rho between distance and LCA bin over the observed pairs (positive = distance
    grows as pairs diverge higher up the tree)."""
    observed = ~np.isnan(x)
    if observed.sum() < 3 or np.unique(bin_index[observed]).size < 2:
        return float("nan")
    return float(-spearman_rho(x[observed], bin_index[observed].astype(float)))


def is_monotone_falling(profile: np.ndarray) -> bool:
    """True if the profile never rises from one bin to the next (NaN bins skipped)."""
    return bool(np.all(np.diff(profile[~np.isnan(profile)]) <= 0))


# ---------------------------------------------------------------------------
# Layout shared by the point estimates and the bootstrap draws
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeBins:
    """The per-pair tree labels, as indices: LCA bin position, stratum position, and each stratum's
    reference (first non-empty) bin."""
    levels: np.ndarray
    bin_index: np.ndarray
    strata: list[str]
    stratum_index: np.ndarray
    stratum_ref: list[int]

    @property
    def n_bins(self) -> int:
        return len(self.levels)

    @property
    def labels(self) -> list[str]:
        return [common.tier_label(int(v)) for v in self.levels]


def tree_bins(parts: Parts) -> TreeBins:
    bins = common.lca_bins(parts)
    levels = np.unique(bins)
    bin_index = np.searchsorted(levels, bins)
    labels = common.branch_strata(parts)
    strata = sorted(np.unique(labels))
    stratum_index = np.searchsorted(strata, labels)
    refs = [int(bin_index[stratum_index == s].min()) for s in range(len(strata))]
    return TreeBins(levels, bin_index, strata, stratum_index, refs)


def _statistic(tb: TreeBins):
    """Bootstrap statistic over (rdm, bin_index, stratum_index): profile, rank slope, stratum profiles."""
    K, S = tb.n_bins, len(tb.strata)

    def stat(x: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
        b, s = b.astype(int), s.astype(int)
        out = np.full(K + 1 + S * K, np.nan)
        out[:K] = normalised_profile(x, b, K)
        out[K] = rank_slope(x, b)
        for k in range(S):
            inside = s == k
            out[K + 1 + k * K:K + 1 + (k + 1) * K] = normalised_profile(x[inside], b[inside], K, tb.stratum_ref[k])
        return out

    return stat


@dataclass(frozen=True)
class RdmDepth:
    """Point estimates and 95% bootstrap CIs for one RDM."""
    profile: np.ndarray
    profile_ci: np.ndarray
    slope: float
    slope_ci: tuple[float, float]
    ratio: float
    ratio_ci: tuple[float, float]
    strata_profile: np.ndarray
    strata_ci: np.ndarray


def rdm_depth(
    name: str, x: np.ndarray, parts: Parts, *, n_boot: int, cache_dir: Path | None, seed: int = 0,
) -> RdmDepth:
    """Depth profile, rank slope, root/deepest ratio and per-stratum profiles of one RDM, with bootstrap CIs."""
    tb = tree_bins(parts)
    K, S = tb.n_bins, len(tb.strata)
    stat = _statistic(tb)
    point = stat(x, tb.bin_index.astype(float), tb.stratum_index.astype(float))
    path = None if cache_dir is None else Path(cache_dir) / f"{name}_n{n_boot}.npz"
    draws = image_bootstrap(
        [x, tb.bin_index.astype(float), tb.stratum_index.astype(float)], stat,
        n_boot=n_boot, seed=seed, cache_path=path, extra_key=("depth_gradient_v1", *tb.strata, *tb.stratum_ref),
    )
    ci = np.array([percentile_ci(draws[:, j]) if np.isfinite(draws[:, j]).any() else (np.nan, np.nan)
                   for j in range(draws.shape[1])])
    ratio_draws = 1.0 / draws[:, K - 1]
    return RdmDepth(
        profile=point[:K], profile_ci=ci[:K], slope=float(point[K]), slope_ci=tuple(ci[K]),
        ratio=float(1.0 / point[K - 1]), ratio_ci=percentile_ci(ratio_draws),
        strata_profile=point[K + 1:].reshape(S, K), strata_ci=ci[K + 1:].reshape(S, K, 2),
    )


def depth_gradient(
    rdms: dict[str, np.ndarray], parts: Parts, *, n_boot: int, cache_dir: Path | None, seed: int = 0,
) -> dict[str, RdmDepth]:
    """rdm_depth() for every RDM."""
    return {n: rdm_depth(n, x, parts, n_boot=n_boot, cache_dir=cache_dir, seed=seed) for n, x in rdms.items()}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def profile_tables(results: dict[str, RdmDepth], parts: Parts) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(profile, lower, upper): one column per RDM plus an `n_pairs` column on `profile`; index = LCA bin."""
    tb = tree_bins(parts)
    index = pd.Index(tb.labels, name="lca_bin")
    profile = pd.DataFrame({k: r.profile for k, r in results.items()}, index=index)
    lower = pd.DataFrame({k: r.profile_ci[:, 0] for k, r in results.items()}, index=index)
    upper = pd.DataFrame({k: r.profile_ci[:, 1] for k, r in results.items()}, index=index)
    profile.insert(0, "n_pairs", np.bincount(tb.bin_index, minlength=tb.n_bins))
    return profile, lower, upper


def monotonicity_table(results: dict[str, RdmDepth]) -> pd.DataFrame:
    """Per RDM: root / deepest ratio (with CI), whether the profile falls monotonically, rank slope (with CI)."""
    rows = [{
        "rdm": k, "root_deepest_ratio": r.ratio, "ratio_low": r.ratio_ci[0], "ratio_high": r.ratio_ci[1],
        "monotone_falling": is_monotone_falling(r.profile),
        "rank_slope": r.slope, "slope_low": r.slope_ci[0], "slope_high": r.slope_ci[1],
    } for k, r in results.items()]
    return pd.DataFrame(rows).set_index("rdm")


def rank_slope_table(
    rdms: dict[str, np.ndarray], *, n_boot: int, cache_dir: Path | None, parts: Parts | None = None, seed: int = 0,
) -> pd.DataFrame:
    """Rank slope (minus Spearman rho between distance and LCA bin, over observed pairs; positive = distance
    grows as pairs diverge higher up the tree) with its image-bootstrap 95% CI, for any dict of condensed
    RDMs of the curated images. Columns: rank_slope, ci_low, ci_high; index = RDM name. Draws are cached
    in `cache_dir` (None disables caching); `parts` defaults to the curated tree."""
    parts = common.tree_parts() if parts is None else parts
    results = depth_gradient(rdms, parts, n_boot=n_boot, cache_dir=cache_dir, seed=seed)
    table = monotonicity_table(results)[["rank_slope", "slope_low", "slope_high"]]
    return table.rename(columns={"slope_low": "ci_low", "slope_high": "ci_high"})


def strata_table(results: dict[str, RdmDepth], parts: Parts) -> pd.DataFrame:
    """Pair counts per stratum (rows) and LCA bin (columns), with each stratum's reference bin."""
    tb = tree_bins(parts)
    counts = np.zeros((len(tb.strata), tb.n_bins), dtype=int)
    np.add.at(counts, (tb.stratum_index, tb.bin_index), 1)
    table = pd.DataFrame(counts, index=pd.Index(tb.strata, name="stratum"), columns=tb.labels)
    table["normalised_to_bin"] = [tb.labels[r] for r in tb.stratum_ref]
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _add_curve(
    fig: go.Figure, name: str, y: np.ndarray, lo: np.ndarray | None, hi: np.ndarray | None,
    *, row: int | None = None, col: int | None = None, show_legend: bool = True,
) -> None:
    where = {} if row is None else {"row": row, "col": col}
    x = list(range(len(y)))
    kwargs = rdm_style.scatter_kwargs(name)
    if lo is not None and not np.all(np.isnan(lo)):
        ok = ~(np.isnan(lo) | np.isnan(hi))
        xs = [i for i, o in zip(x, ok) if o]
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=list(hi[ok]) + list(lo[ok])[::-1], fill="toself", mode="none",
            fillcolor=_rgba(rdm_style.style(name).color, 0.15), legendgroup=kwargs["legendgroup"],
            showlegend=False, hoverinfo="skip",
        ), **where)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", showlegend=show_legend, **kwargs), **where)


def plot_depth_profile(
    profile: pd.DataFrame, lower: pd.DataFrame | None = None, upper: pd.DataFrame | None = None,
    *, title: str | None = None,
) -> go.Figure:
    """Normalised mean distance per LCA bin, one curve per RDM (profile minus its `n_pairs` column), with
    optional bootstrap bands. The tick labels carry the number of pairs in each bin."""
    names = rdm_style.legend_order([c for c in profile.columns if c != "n_pairs"])
    fig = go.Figure()
    for name in names:
        band = (None, None) if lower is None else (lower[name].to_numpy(), upper[name].to_numpy())
        _add_curve(fig, name, profile[name].to_numpy(), *band)
    ticks = [f"{lab}<br><sub>n={int(n):,}</sub>" for lab, n in zip(profile.index, profile["n_pairs"])]
    fig.update_layout(
        title=title or "Distance vs hierarchical separation"
                       "<br><sub>mean distance per LCA bin, relative to the root bin; shaded = 95% image bootstrap</sub>",
        xaxis=dict(title="LCA depth (0 = diverge at the root, higher = stay together deeper)",
                   tickmode="array", tickvals=list(range(len(ticks))), ticktext=ticks),
        yaxis=dict(title="mean distance, relative to bin 0"),
        template=_TEMPLATE, height=520, width=900, legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
    )
    return fig


def plot_rank_slopes(table: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """Rank slope per RDM with its bootstrap CI (columns rank_slope, and slope_low/slope_high or ci_low/ci_high)."""
    low = "slope_low" if "slope_low" in table else "ci_low"
    high = "slope_high" if "slope_high" in table else "ci_high"
    names = rdm_style.legend_order(list(table.index))
    fig = go.Figure()
    for name in names:
        row = table.loc[name]
        kwargs = rdm_style.scatter_kwargs(name)
        fig.add_trace(go.Scatter(
            x=[rdm_style.tick_label(name)], y=[row["rank_slope"]], mode="markers", showlegend=False,
            error_y=dict(type="data", symmetric=False, array=[row[high] - row["rank_slope"]],
                         arrayminus=[row["rank_slope"] - row[low]], color=kwargs["marker"]["color"]),
            name=kwargs["name"], marker=kwargs["marker"],
        ))
    fig.update_layout(
        title=title or "Rank slope of distance on LCA bin<br><sub>positive = distance grows as pairs diverge "
                       "higher up the tree; 95% image bootstrap</sub>",
        yaxis=dict(title="rank slope (-Spearman rho)"), xaxis=dict(categoryorder="array", categoryarray=[rdm_style.tick_label(n) for n in names]),
        template=_TEMPLATE, height=460, width=900,
    )
    return fig


def plot_strata_profiles(results: dict[str, RdmDepth], parts: Parts, *, title: str | None = None) -> go.Figure:
    """One subplot per branch-depth stratum (shared y), each curve normalised to the stratum's first non-empty bin."""
    tb = tree_bins(parts)
    counts = strata_table(results, parts)
    titles = [f"{s} ({int(counts.loc[s, tb.labels].sum()):,} pairs)" for s in tb.strata]
    fig = make_subplots(rows=1, cols=len(tb.strata), shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.04)
    for name in rdm_style.legend_order(list(results)):
        r = results[name]
        for k in range(len(tb.strata)):
            empty = counts.loc[tb.strata[k], tb.labels].to_numpy() == 0
            y = np.where(empty, np.nan, r.strata_profile[k])
            lo, hi = r.strata_ci[k, :, 0], r.strata_ci[k, :, 1]
            _add_curve(fig, name, y, np.where(empty, np.nan, lo), np.where(empty, np.nan, hi),
                       row=1, col=k + 1, show_legend=(k == 0))
    ticks = dict(tickmode="array", tickvals=list(range(tb.n_bins)), ticktext=tb.labels)
    fig.update_xaxes(title_text="LCA depth", **ticks)
    fig.update_yaxes(title_text="mean distance, relative to the stratum's first non-empty bin", col=1)
    fig.update_layout(
        title=title or "Depth profile within each branch-depth stratum", template=_TEMPLATE,
        height=480, width=1100, legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
    )
    return fig
