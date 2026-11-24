"""
How much rank structure does each RDM share with the tree, and with each other?
(2_rank_agreement.ipynb is the front end.)

Agreement is a rank correlation over the pairs two RDMs both observe (behav_joint has ~13% unobserved
pairs). Kendall tau-a is primary for agreement with sem_km, which has only nine distinct values and so
many ties: tau-a keeps the ties in the denominator, so a perfectly monotone model still falls short of 1.
Spearman is reported alongside.

Uncertainty and inference respect that RDM entries are not independent: confidence intervals come from
an image bootstrap (images are the sampling unit), tests against chance from an image-relabeling
permutation (utils.stats.label_shuffle_test). Differences between two RDMs' agreement with the tree are
tested with a paired image bootstrap: both statistics are computed on the same resampled images, and on
the pairs all three RDMs observe, so the difference cannot come from different pairs or different images.

Behavioral RDMs are also read against their own ceiling. A model cannot correlate with a noisy RDM
better than sqrt(r_SB) (r_SB = the Spearman-Brown-projected split-half reliability of that RDM), so a
behavioral RDM's Spearman with the tree is reported as a fraction of that bound. The bound is derived
for the raw RDM; for the MDS embedding, whose noise is partly averaged out, it is only indicative.

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import rank_agreement as ra
    ra.correlation_matrix(rdms, "kendall_a")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests

from analysis.s2_semantic_hierarchy import common as c
from analysis.utils import plotting
from analysis.utils.stats import image_bootstrap, label_shuffle_test, percentile_ci

_TEMPLATE = "plotly_white"
N_BOOT = 1000
N_PERM = 1000
SEED = 0


# ---------------------------------------------------------------------------
# 2.1 Pairwise agreement between all RDMs
# ---------------------------------------------------------------------------


def correlation_matrix(rdms: dict[str, np.ndarray], stat: str = "kendall_a") -> pd.DataFrame:
    """Square DataFrame of `stat` between every pair of RDMs, each over its pairwise-complete pairs."""
    keys = list(rdms)
    m = np.eye(len(keys))
    for i, a in enumerate(keys):
        for j in range(i + 1, len(keys)):
            m[i, j] = m[j, i] = c.rank_agreement(rdms[a], rdms[keys[j]], (stat,))[stat]
    return pd.DataFrame(m, index=keys, columns=keys)


def plot_correlation_matrix(corr: pd.DataFrame, *, stat: str = "kendall_a", title: str | None = None) -> go.Figure:
    """Heatmap of rank agreement between RDMs. Answers: what shares rank structure?"""
    names = plotting.rdm_legend_order(list(corr.columns))
    corr = corr.loc[names, names]
    ticks = [plotting.rdm_tick_label(n) for n in names]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=ticks, y=ticks, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).to_numpy(), texttemplate="%{text}", textfont={"size": 12},
        colorbar={"title": c.STAT_LABEL[stat]},
    ))
    fig.update_layout(
        title=title or f"Shared rank structure between RDMs ({c.STAT_LABEL[stat]}, pairwise-complete pairs)",
        template=_TEMPLATE, height=600, width=760, yaxis={"autorange": "reversed"},
    )
    return fig


# ---------------------------------------------------------------------------
# 2.2 Agreement of every RDM with the tree
# ---------------------------------------------------------------------------


def agreement_with_reference(
    rdms: dict[str, np.ndarray], *, reference: str = c.REFERENCE, n_boot: int = N_BOOT, n_perm: int = N_PERM,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per RDM (index): point estimate, image-bootstrap 95% CI and label-shuffle p, for each rank statistic."""
    ref = rdms[reference]
    rows = []
    for name, rdm in rdms.items():
        if name == reference:
            continue
        draws = image_bootstrap([rdm, ref], c.rank_agreement_list, n_boot=n_boot, seed=seed,
                                cache_path=c.cache_path(f"rank_boot_{name}_{n_boot}.npz"))
        shuffle = label_shuffle_test(rdm, ref, statistics=tuple(c.RANK_STATS), n_perm=n_perm, seed=seed,
                                     cache_path=c.cache_path(f"rank_perm_{name}_{n_perm}.npz"))
        row: dict[str, float] = {"n_pairs": int(c.pairwise_complete(rdm, ref).sum())}
        for j, stat in enumerate(c.RANK_STATS):
            lo, hi = percentile_ci(draws[:, j])
            row |= {stat: shuffle.observed[stat], f"{stat}_lo": lo, f"{stat}_hi": hi,
                    f"{stat}_p": shuffle.p_value[stat]}
        rows.append({"rdm": name, **row})
    table = pd.DataFrame(rows).set_index("rdm")
    return table.loc[plotting.rdm_legend_order(list(table.index))]


def plot_forest(table: pd.DataFrame, *, stat: str = "kendall_a", title: str | None = None) -> go.Figure:
    """One row per RDM: agreement with the tree as a dot, 95% bootstrap CI as error bars."""
    fig = go.Figure()
    for name in table.index:
        row = table.loc[name]
        fig.add_trace(go.Scatter(
            x=[row[stat]], y=[plotting.rdm_label(name)], mode="markers",
            error_x={"type": "data", "symmetric": False, "array": [row[f"{stat}_hi"] - row[stat]],
                     "arrayminus": [row[stat] - row[f"{stat}_lo"]], "color": plotting.rdm_style(name).color},
            **{k: v for k, v in plotting.rdm_scatter_kwargs(name, size=11).items() if k != "line"},
        ))
    fig.add_vline(x=0, line={"color": "grey", "dash": "dot"})
    fig.update_layout(
        title=title or f"Agreement with the KM tree ({c.STAT_LABEL[stat]}, 95% image-bootstrap CI)",
        xaxis={"title": c.STAT_LABEL[stat], "zeroline": False}, yaxis={"autorange": "reversed"},
        template=_TEMPLATE, height=420, width=820, showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 2.3 Behavioral RDMs against their ceiling
# ---------------------------------------------------------------------------


def ceiling_table(table: pd.DataFrame, bracket: dict[str, float], names: list[str]) -> pd.DataFrame:
    """Spearman with the tree of the behavioral RDMs in `names`, as a fraction of both ends of the raw
    RDM's noise-free-model ceiling bracket (joint_behavior.ceiling_bracket: `low` from the empirical
    half-sample reliability, `high` from the Spearman-Brown projection)."""
    rows = [{"rdm": n, "spearman": table.loc[n, "spearman"], "kendall_a": table.loc[n, "kendall_a"],
             "of_low": table.loc[n, "spearman"] / bracket["low"], "of_high": table.loc[n, "spearman"] / bracket["high"]}
            for n in names if n in table.index]
    return pd.DataFrame(rows).set_index("rdm")


# ---------------------------------------------------------------------------
# 2.4 Paired differences
# ---------------------------------------------------------------------------


def _difference(a: np.ndarray, b: np.ndarray, ref: np.ndarray) -> list[float]:
    """Agreement of `a` minus agreement of `b` with `ref`, per rank statistic, on the pairs all three observe."""
    mask = c.pairwise_complete(a, b, ref)
    agree_a = c.rank_agreement_list(a[mask], ref[mask])
    agree_b = c.rank_agreement_list(b[mask], ref[mask])
    return [x - y for x, y in zip(agree_a, agree_b)]


def paired_difference(
    a: np.ndarray, b: np.ndarray, ref: np.ndarray, *, n_boot: int = N_BOOT, seed: int = SEED, tag: str = "",
) -> dict[str, float]:
    """Paired image bootstrap of agree(a, ref) - agree(b, ref): estimate, 95% CI and two-sided p per statistic.

    p = 2 * min(P(diff <= 0), P(diff >= 0)) over the draws, floored at 1 / n_boot.
    """
    cache = c.cache_path(f"rank_diff_{tag}_{n_boot}.npz") if tag else None
    draws = image_bootstrap([a, b, ref], _difference, n_boot=n_boot, seed=seed, cache_path=cache)
    estimate = _difference(a, b, ref)
    out: dict[str, float] = {"n_pairs": int(c.pairwise_complete(a, b, ref).sum())}
    for j, stat in enumerate(c.RANK_STATS):
        d = draws[:, j]
        lo, hi = percentile_ci(d)
        p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
        out |= {stat: estimate[j], f"{stat}_lo": lo, f"{stat}_hi": hi, f"{stat}_p": float(max(min(p, 1.0), 1 / n_boot))}
    return out


def paired_family(
    rdms: dict[str, np.ndarray], pairs: list[tuple[str, str]], *, reference: str = c.REFERENCE,
    n_boot: int = N_BOOT, seed: int = SEED, stat: str = "kendall_a",
) -> pd.DataFrame:
    """Paired differences for each (a, b) in `pairs`, with BH-FDR over the family on `stat`'s p-values."""
    rows = [{"a": a, "b": b, **paired_difference(rdms[a], rdms[b], rdms[reference], n_boot=n_boot, seed=seed,
                                                  tag=f"{a}_vs_{b}")} for a, b in pairs]
    table = pd.DataFrame(rows)
    table["p_fdr"] = multipletests(table[f"{stat}_p"], method="fdr_bh")[1]
    return table.set_index(["a", "b"])


# ---------------------------------------------------------------------------
# 2.5 Sensitivity: identical pairs for every RDM
# ---------------------------------------------------------------------------


def agreement_on_pairs_of(
    rdms: dict[str, np.ndarray], anchor: str, *, reference: str = c.REFERENCE,
) -> pd.DataFrame:
    """Rank agreement with `reference` for every RDM restricted to the pairs `anchor` observed."""
    mask = c.pairwise_complete(rdms[anchor], rdms[reference])
    rows = [{"rdm": n, "n_pairs": int(mask.sum()), **c.rank_agreement(np.where(mask, r, np.nan), rdms[reference])}
            for n, r in rdms.items() if n != reference]
    table = pd.DataFrame(rows).set_index("rdm")
    return table.loc[plotting.rdm_legend_order(list(table.index))]
