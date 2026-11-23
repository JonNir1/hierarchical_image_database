"""
Which SMACOF embeddings of the joint behavioral RDM are carried into notebooks 2 to 6, and does embedding
move the behavioral RDM toward the tree? (1_mds_dimensionality.ipynb is the front end.)

Both dimensionalities are chosen without looking at the tree: choosing the d whose embedding best matches
sem_km and then reporting that agreement would be circular. Two embeddings are kept, because no single d
is privileged (held-out prediction keeps rising slowly up to the largest d fitted):

  low  (behav_joint_mds_low):  the elbow of the stress curve, by the kneedle rule (Satopaa et al., 2011):
       the d farthest below the chord joining the first and last points of the normalised curve.
  high (behav_joint_mds_high): the smallest d capturing HIGH_GAIN (95%) of the held-out prediction gain
       of the best d over the raw half-RDM. Held-out prediction: participants are split into halves, each
       half's raw RDM is embedded at every d, and the embedding is scored by how well its distances predict
       the *other* half's raw ratings.

Agreement with the tree is then reported for every d, descriptively, against two references: the raw RDM
and the highest Spearman any noise-free model could reach given the raw RDM's own reliability,
sqrt(r_SB), where r_SB is the Spearman-Brown-projected split-half reliability.

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import mds_dimensionality as md
    fits = md.sweep(behav)
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.s0_build_rdms.behavior.noise_ceiling import (
    _DEFAULT_CACHE_DIR, _parse_trials_by_participant, _spearman_brown, make_splits,
)
from analysis.s0_build_rdms.behavior.spam_ratings import JOINT_VARIANT, _combine, _select_participants
from analysis.s0_build_rdms.common import image_manifest_index, save_rdm
from analysis.s2_semantic_hierarchy import common as c
from analysis.utils import rdm_style
from analysis.utils.disk_cache import cached_array, cached_frame, fingerprint
from analysis.utils.hierarchy import Parts
from analysis.utils.load_data import load_data
from analysis.utils.mds import run_mds_sweep

AGGREGATION = "simple"
MDS_DIMS = (*range(1, 16), 20, 25, 30)
SPLIT_DIMS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)
N_SPLITS_MDS = 5
HIGH_GAIN = 0.95
N_SPLITS_CEILING = 100
SEED = 0
MDS_CACHE_DIR = c.cache_path("mds")
_TEMPLATE = "plotly_white"
SWEEP_STYLE = "behav_joint_mds"  # style of curves over d (an embedding without a chosen tier)

TrialsByParticipant = dict[str, list[tuple[np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# Data: the pooled participants' trials, and the full and half RDMs built from them
# ---------------------------------------------------------------------------


def joint_trials() -> tuple[TrialsByParticipant, int]:
    """Every retained participant of both cohorts, with their parsed non-catch trials (as in behav_joint)."""
    data = load_data(_DEFAULT_CACHE_DIR, prod_only=True)
    ids = list(_select_participants(data["participants"], JOINT_VARIANT)["participant_id"])
    image_to_idx, n_conds = image_manifest_index()
    return _parse_trials_by_participant(data["trials"], ids, JOINT_VARIANT, image_to_idx, n_conds), n_conds


def combine(trials_by_participant: TrialsByParticipant, ids: list[str], n_conds: int) -> tuple[np.ndarray, np.ndarray]:
    """('simple' RDM, ratings per pair) of the participants `ids`."""
    return _combine([t for pid in ids for t in trials_by_participant[pid]], n_conds, AGGREGATION)


def split_halves(
    trials_by_participant: TrialsByParticipant, n_conds: int, *, n_splits: int, seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Raw RDMs of the two participant halves of each of `n_splits` random splits (noise_ceiling.make_splits)."""
    splits = make_splits(list(trials_by_participant), n_splits, seed)
    return [(combine(trials_by_participant, a, n_conds)[0], combine(trials_by_participant, b, n_conds)[0])
            for a, b in splits]


def coverage(counts: np.ndarray, parts: Parts) -> pd.DataFrame:
    """Fraction of pairs observed and ratings per observed pair: overall, per pooled LCA bin, per stratum."""
    observed = counts > 0
    groups = {"overall": np.full(len(counts), "all", dtype=object),
              "LCA bin": np.array([c.tier_label(int(b)) for b in c.lca_bins(parts)], dtype=object),
              "stratum": c.branch_strata(parts)}
    rows = []
    for grouping, labels in groups.items():
        for value in sorted(pd.unique(labels), key=str):  # str order puts ">=4" after the digits
            sel = labels == value
            rows.append({"grouping": grouping, "group": value, "n_pairs": int(sel.sum()),
                         "frac_observed": float(observed[sel].mean()),
                         "ratings_per_observed_pair": float(counts[sel & observed].mean())})
    return pd.DataFrame(rows).set_index(["grouping", "group"])


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def sweep(behav: np.ndarray, dims: tuple[int, ...] = MDS_DIMS) -> dict[int, dict]:
    """SMACOF fits of the full joint RDM at every d (cached per d)."""
    return run_mds_sweep(behav, list(dims), f"behav_joint_{fingerprint(behav)[:10]}", MDS_CACHE_DIR)


def stress_table(fits: dict[int, dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ndim": d, "stress": f["stress"], "niter": f["niter"], "converged": not f["needs_more_iters"]}
         for d, f in sorted(fits.items())]
    ).set_index("ndim")


def _fit_one(task: tuple[np.ndarray, int]) -> np.ndarray:
    rdm, d = task
    return run_mds_sweep(rdm, [d], f"half_{fingerprint(rdm)[:10]}", MDS_CACHE_DIR)[d]["confdist"]


def fit_halves(
    halves: list[tuple[np.ndarray, np.ndarray]], dims: tuple[int, ...] = SPLIT_DIMS, *, n_jobs: int = 6,
) -> dict[tuple[int, int, int], np.ndarray]:
    """{(split, half, d): embedded distances} for every half RDM at every d, fitted in parallel, cached per fit."""
    keys = [(s, h, d) for s in range(len(halves)) for h in (0, 1) for d in dims]
    tasks = [(halves[s][h], d) for s, h, d in keys]
    if n_jobs == 1:
        return dict(zip(keys, map(_fit_one, tasks)))
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        return dict(zip(keys, executor.map(_fit_one, tasks)))


# ---------------------------------------------------------------------------
# Choosing d* without the tree
# ---------------------------------------------------------------------------


def heldout_prediction(
    halves: list[tuple[np.ndarray, np.ndarray]], fits: dict[tuple[int, int, int], np.ndarray],
) -> pd.DataFrame:
    """Rank agreement of each half's embedding with the other half's raw ratings (on that half's observed pairs).

    One row per (split, direction, d); 'raw' rows (d = 0) score the fitting half's raw RDM instead of its
    embedding, i.e. the plain split-half reliability on the same pairs.
    """
    rows = []
    for (s, h, d), confdist in fits.items():
        target = halves[s][1 - h]
        rows.append({"split": s, "fit_half": h, "ndim": d, **c.rank_agreement(confdist, target)})
    for s, (a, b) in enumerate(halves):
        for h, (fit, target) in enumerate(((a, b), (b, a))):
            rows.append({"split": s, "fit_half": h, "ndim": 0, **c.rank_agreement(fit, target)})
    return pd.DataFrame(rows)


def heldout_summary(heldout: pd.DataFrame, stat: str = "spearman") -> pd.DataFrame:
    """Mean and SE across splits (directions averaged within a split) of the held-out statistic per d."""
    per_split = heldout.groupby(["ndim", "split"])[stat].mean().unstack("split")
    return pd.DataFrame({"mean": per_split.mean(axis=1),
                         "se": per_split.std(axis=1, ddof=1) / np.sqrt(per_split.shape[1])})


def choose_high(summary: pd.DataFrame, gain: float = 0.95) -> int:
    """Smallest d whose held-out prediction captures `gain` of the best d's improvement over the raw row (d = 0)."""
    raw = summary.at[0, "mean"]
    emb = summary.drop(index=0)
    threshold = raw + gain * (emb["mean"].max() - raw)
    return int(emb.index[emb["mean"] >= threshold].min())


def choose_low(stress: pd.DataFrame) -> int:
    """Kneedle elbow of the stress curve: the d farthest below the chord of the curve normalised to [0, 1]^2."""
    d = stress.index.to_numpy(dtype=float)
    y = stress["stress"].to_numpy()
    x_n = (d - d[0]) / (d[-1] - d[0])
    y_n = (y - y[-1]) / (y[0] - y[-1])
    return int(stress.index[int(np.argmax((1.0 - x_n) - y_n))])


def raw_ceiling(trials_by_participant: TrialsByParticipant, n_conds: int, *, n_splits: int = N_SPLITS_CEILING,
                seed: int = SEED) -> pd.DataFrame:
    """Split-half reliability of the raw joint RDM per split, with its Spearman-Brown projection (cached)."""
    def run() -> pd.DataFrame:
        rows = []
        for s, (a, b) in enumerate(split_halves(trials_by_participant, n_conds, n_splits=n_splits, seed=seed)):
            stats = c.rank_agreement(a, b)
            rows.append({"split": s, **stats, "spearman_brown": _spearman_brown(stats["spearman"])})
        return pd.DataFrame(rows)

    key = fingerprint(sorted(trials_by_participant), n_conds, n_splits, seed, AGGREGATION)
    return cached_frame(c.cache_path("raw_ceiling.pkl"), key, run)


def attenuation_bound(r_sb: float) -> float:
    """Highest Spearman a noise-free model can reach with an RDM of split-half reliability r_SB (sqrt r_SB)."""
    return float(np.sqrt(r_sb))


# ---------------------------------------------------------------------------
# Agreement with the tree, per d (descriptive)
# ---------------------------------------------------------------------------


def km_agreement_by_dim(fits: dict[int, dict], behav: np.ndarray, km: np.ndarray) -> pd.DataFrame:
    """Rank agreement with sem_km per d, on the raw RDM's observed pairs and on all pairs; d = 0 is the raw RDM."""
    observed = ~np.isnan(behav)
    rows = [{"ndim": 0, "pairs": "observed", **c.rank_agreement(behav, km)}]
    for d, fit in sorted(fits.items()):
        rows.append({"ndim": d, "pairs": "observed", **c.rank_agreement(np.where(observed, fit["confdist"], np.nan), km)})
        rows.append({"ndim": d, "pairs": "all", **c.rank_agreement(fit["confdist"], km)})
    return pd.DataFrame(rows)


def save_chosen(fits: dict[int, dict], name: str, d: int, *, rule: str) -> None:
    """Persist the embedding at `d` as D_<name>.npy, with its provenance in metadata.json."""
    fit = fits[d]
    save_rdm(name, fit["confdist"], metric="smacof_euclidean",
             source="analysis/s2_semantic_hierarchy/mds_dimensionality.py",
             extra={"from_rdm": c.RDM_FILE[c.BEHAV], "ndim": d, "stress": fit["stress"],
                    "niter": fit["niter"], "selection_rule": rule})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


Chosen = dict[str, int]  # {c.BEHAV_MDS_LOW: d, c.BEHAV_MDS_HIGH: d}


def _d_line(fig: go.Figure, chosen: Chosen, **where) -> None:
    for name, d in chosen.items():
        s = rdm_style.style(name)
        fig.add_vline(x=d, line={"color": s.color, "dash": s.dash, "width": 1.5},
                      annotation_text=f"{s.tier} d = {d}", annotation_position="top right", **where)


def fig_coverage(table: pd.DataFrame) -> go.Figure:
    """Fraction of pairs observed per pooled LCA bin and per stratum, with the overall fraction as a line."""
    overall = float(table.loc[("overall", "all"), "frac_observed"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=["per LCA bin", "per branch-depth stratum"],
                        shared_yaxes=True, horizontal_spacing=0.08)
    s = rdm_style.style(c.BEHAV)
    for col, grouping in enumerate(("LCA bin", "stratum"), start=1):
        part = table.loc[grouping]
        fig.add_trace(go.Bar(x=list(part.index), y=part["frac_observed"], marker_color=s.color,
                             text=[f"{v:.1%}" for v in part["frac_observed"]], textposition="outside",
                             showlegend=False), row=1, col=col)
        fig.add_hline(y=overall, line={"color": s.color, "dash": "dot", "width": 1}, row=1, col=col)
    fig.update_yaxes(title_text="fraction of pairs observed", range=[0, 1.08], row=1, col=1)
    fig.update_layout(title=f"Pair coverage of the joint behavioral RDM (dotted: overall {overall:.1%})",
                      template=_TEMPLATE, height=420, width=900)
    return fig


def fig_stress(stress: pd.DataFrame, chosen: Chosen) -> go.Figure:
    fig = go.Figure(go.Scatter(x=stress.index, y=stress["stress"], mode="lines+markers",
                               **rdm_style.scatter_kwargs(SWEEP_STYLE)))
    _d_line(fig, chosen)
    fig.update_layout(title="Stress-1 of the SMACOF embedding of the joint behavioral RDM",
                      xaxis_title="MDS dimensionality d", yaxis_title="stress-1",
                      template=_TEMPLATE, height=420, width=760, showlegend=False)
    return fig


def fig_heldout(summary: pd.DataFrame, chosen: Chosen, *, stat_label: str = "Spearman rho") -> go.Figure:
    """Held-out prediction per d (mean +/- SE across splits), with the raw half-vs-half value as a line."""
    emb = summary.drop(index=0)
    kw = rdm_style.scatter_kwargs(SWEEP_STYLE)
    fig = go.Figure(go.Scatter(x=emb.index, y=emb["mean"], mode="lines+markers",
                               error_y={"type": "data", "array": emb["se"], "visible": True}, **kw))
    raw = rdm_style.style(c.BEHAV)
    fig.add_hline(y=summary.at[0, "mean"], line={"color": raw.color, "width": 1.5},
                  annotation_text="raw half vs raw half", annotation_position="bottom right")
    _d_line(fig, chosen)
    fig.update_layout(title="Held-out prediction: one half's embedding vs the other half's raw ratings",
                      xaxis_title="MDS dimensionality d", yaxis_title=f"{stat_label} (mean +/- SE over splits)",
                      template=_TEMPLATE, height=440, width=760, showlegend=False)
    return fig


def fig_km_by_dim(table: pd.DataFrame, chosen: Chosen, *, ceiling: float | None, stat: str = "spearman") -> go.Figure:
    """Agreement with sem_km per d (observed and all pairs), raw RDM and attenuation ceiling as lines."""
    fig = go.Figure()
    raw = table.loc[table["ndim"] == 0, stat].iloc[0]
    for pairs, dash in (("observed", "solid"), ("all", "dot")):
        part = table[(table["ndim"] > 0) & (table["pairs"] == pairs)]
        kw = rdm_style.scatter_kwargs(SWEEP_STYLE)
        kw["line"]["dash"] = dash
        kw["name"] = f"embedding, {pairs} pairs"
        fig.add_trace(go.Scatter(x=part["ndim"], y=part[stat], mode="lines+markers", **kw))
    colour = rdm_style.style(c.BEHAV).color
    fig.add_hline(y=raw, line={"color": colour, "width": 1.5}, annotation_text="raw RDM",
                  annotation_position="bottom right")
    if ceiling is not None:
        fig.add_hline(y=ceiling, line={"color": colour, "width": 1.5, "dash": "dot"},
                      annotation_text="noise-free-model ceiling", annotation_position="top right")
    _d_line(fig, chosen)
    fig.update_layout(title=f"Agreement of the behavioral embedding with the KM tree ({c.STAT_LABEL[stat]})",
                      xaxis_title="MDS dimensionality d", yaxis_title=c.STAT_LABEL[stat],
                      template=_TEMPLATE, height=460, width=800,
                      legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"})
    return fig


def fig_robustness(slopes: pd.DataFrame, nested: pd.DataFrame, chosen: Chosen) -> go.Figure:
    """Rank slope (notebook 3) and nested per-level d (notebook 4) of the behavioral embedding per d.

    `slopes` is depth_gradient.rank_slope_table() and `nested` level_separation.nested_separation_table()
    over {behav_joint, behav_joint_mds<d>...}; the raw RDM is drawn as a horizontal line in each panel.
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["depth-gradient rank slope", "nested separation d per level"])
    raw = rdm_style.style(c.BEHAV)
    emb = slopes.drop(index=c.BEHAV)
    dims = [rdm_style.parse(n)[3] for n in emb.index]
    fig.add_trace(go.Scatter(
        x=dims, y=emb["rank_slope"], mode="lines+markers", showlegend=False,
        error_y={"type": "data", "symmetric": False, "array": emb["ci_high"] - emb["rank_slope"],
                 "arrayminus": emb["rank_slope"] - emb["ci_low"]},
        **rdm_style.scatter_kwargs(SWEEP_STYLE)), row=1, col=1)
    fig.add_hline(y=slopes.at[c.BEHAV, "rank_slope"], line={"color": raw.color, "width": 1.5}, row=1, col=1)
    for k, level in enumerate(sorted(nested["level"].unique())):
        part = nested[(nested["level"] == level) & (nested["rdm"] != c.BEHAV)]
        kw = rdm_style.scatter_kwargs(SWEEP_STYLE, lighten=0.3 * k)
        kw["name"] = f"L{level}: {c.LEVEL_NAMES[level]}"
        kw["legendgroup"] = f"L{level}"
        fig.add_trace(go.Scatter(
            x=[rdm_style.parse(n)[3] for n in part["rdm"]], y=part["d"], mode="lines+markers",
            error_y={"type": "data", "symmetric": False, "array": part["ci_hi"] - part["d"],
                     "arrayminus": part["d"] - part["ci_lo"]}, **kw), row=1, col=2)
        raw_d = nested.loc[(nested["level"] == level) & (nested["rdm"] == c.BEHAV), "d"].iloc[0]
        fig.add_hline(y=raw_d, line={"color": kw["line"]["color"], "width": 1}, row=1, col=2)
    for col in (1, 2):
        _d_line(fig, chosen, row=1, col=col)
        fig.update_xaxes(title_text="MDS dimensionality d", row=1, col=col)
    fig.update_layout(title="Behavioral embedding across d (horizontal lines: the raw RDM)",
                      template=_TEMPLATE, height=460, width=1050,
                      legend={"orientation": "h", "y": -0.25, "x": 0.75, "xanchor": "center"})
    return fig
