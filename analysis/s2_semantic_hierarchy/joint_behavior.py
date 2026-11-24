"""
The joint behavioral RDM on its own terms (0b_joint_behavior.ipynb is the front end): where its data are
missing, how noisy it is, how it relates to external data, and how it decomposes into semantic and
sensory parts. Observational, like 0a_tree_shape: it describes `behav_joint` before notebooks 1 to 6
compare it with the tree.

`behav_joint` pools both SHINE cohorts (one shown the original images, one the SHINE-processed ones).
Every participant-level quantity therefore pools the 155 retained participants, and stimulus-derived
predictors that differ between the variants (pixels, CLIP) enter as the mean of their pre- and post-SHINE
RDMs, matching the near-even pooling (79 pre, 76 post).

Noise model. Three views of the ceiling, because they answer different questions:

  Nili et al. (2014) bounds on a *per-participant* model fit: the lower bound is each participant's RDM
      against the combined RDM of everyone else (leave one out), the upper bound against the combined RDM
      of everyone including that participant. On sparse SpAM data a participant's own ratings are a large
      share of each entry they rated (reported as `self_share`), which inflates the upper bound.
  Agreement by group size: two disjoint random groups of n participants, for several n. This is the
      empirical reliability of a group RDM of size n, the relevant ceiling for a *group* RDM's agreement
      with a model.
  Split-half with the Spearman-Brown projection to the full sample (the earlier ceiling). It overstates
      what full-size samples reach here, because extra participants mostly add newly covered pairs rather
      than more ratings per pair. The full-sample group ceiling is therefore bracketed: at least the
      half-sample value, at most the Spearman-Brown value.

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import joint_behavior as jb
    trials, n_conds = jb.joint_trials()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from scipy.spatial.distance import squareform

from analysis.s0_build_rdms.behavior.noise_ceiling import (
    _DEFAULT_CACHE_DIR, _parse_trials_by_participant, _spearman_brown, group_size_reliability, make_splits,
    nili_upper_reliability,
)
from analysis.s0_build_rdms.behavior.spam_ratings import JOINT_VARIANT, _combine, _select_participants
from analysis.s0_build_rdms.common import image_manifest_index, load_manifest, load_rdm
from analysis.s1_pre_vs_post_shine import decomposition as dec
from analysis.s1_pre_vs_post_shine.demographics_and_behavior import main_trials
from analysis.s1_pre_vs_post_shine.reference_agreement import _digest, loo_model_fit
from analysis.s2_semantic_hierarchy import common as c
from analysis.utils import plotting
from analysis.utils.disk_cache import cached_frame, fingerprint
from analysis.utils.hierarchy import Parts
from analysis.utils.load_data import load_data
from analysis.utils.stats import percentile_ci

AGGREGATION = "simple"
SEED = 0
N_SPLITS_CEILING = 100
GROUP_SIZES = (10, 20, 38, 77)
N_SPLITS_GROUP = 20
CATEGORY_ORDER = ("HF", "HB", "AF", "AB", "NO", "HO")
CATEGORY_LABEL = {
    "HF": "human face", "HB": "human body", "AF": "animal face", "AB": "animal body",
    "NO": "natural object", "HO": "handmade object",
}
JOINT_MODELS: dict[str, tuple[str, ...]] = {m: dec.MODELS[m] for m in ("km", "km+clip", "km+wn+clip")}
_TEMPLATE = "plotly_white"

TrialsByParticipant = dict[str, list[tuple[np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def joint_participants() -> tuple[pd.DataFrame, list[str]]:
    """(trials table, ids) of every retained participant of both cohorts, the participants of behav_joint."""
    data = load_data(_DEFAULT_CACHE_DIR, prod_only=True)
    ids = list(_select_participants(data["participants"], JOINT_VARIANT)["participant_id"])
    return data["trials"], ids


def joint_trials() -> tuple[TrialsByParticipant, int]:
    """Every retained participant of both cohorts, with their parsed non-catch trials (as in behav_joint)."""
    df_trials, ids = joint_participants()
    image_to_idx, n_conds = image_manifest_index()
    return _parse_trials_by_participant(df_trials, ids, JOINT_VARIANT, image_to_idx, n_conds), n_conds


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


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


def image_categories() -> np.ndarray:
    """The six-way image category (manifest `category`: HF, HB, AF, AB, NO, HO) in manifest order."""
    return load_manifest()["category"].to_numpy()


def category_pairs(categories: np.ndarray) -> np.ndarray:
    """Condensed unordered category-pair label per pair, e.g. 'HF-HB', in CATEGORY_ORDER."""
    rank = {code: i for i, code in enumerate(CATEGORY_ORDER)}
    codes = np.array([rank[x] for x in categories])
    lo = squareform(np.minimum(codes[:, None], codes[None, :]), checks=False)
    hi = squareform(np.maximum(codes[:, None], codes[None, :]), checks=False)
    names = np.array(CATEGORY_ORDER, dtype=object)
    return names[lo] + "-" + names[hi]


def coverage(counts: np.ndarray, parts: Parts, categories: np.ndarray | None = None) -> pd.DataFrame:
    """Fraction of pairs observed and ratings per observed pair: overall, per pooled LCA bin, per stratum,
    and (with `categories`) per unordered category pair."""
    observed = counts > 0
    groups = {"overall": np.full(len(counts), "all", dtype=object),
              "LCA bin": np.array([c.tier_label(int(b)) for b in c.lca_bins(parts)], dtype=object),
              "stratum": c.branch_strata(parts)}
    if categories is not None:
        groups["category pair"] = category_pairs(categories)
    rows = []
    for grouping, labels in groups.items():
        for value in sorted(pd.unique(labels), key=str):  # str order puts ">=4" after the digits
            sel = labels == value
            rows.append({"grouping": grouping, "group": value, "n_pairs": int(sel.sum()),
                         "frac_observed": float(observed[sel].mean()),
                         "ratings_per_observed_pair": float(counts[sel & observed].mean())})
    return pd.DataFrame(rows).set_index(["grouping", "group"])


def category_matrix(table: pd.DataFrame, column: str = "frac_observed") -> pd.DataFrame:
    """coverage()'s category-pair rows as a symmetric 6 x 6 matrix in CATEGORY_ORDER."""
    part = table.loc["category pair", column]
    matrix = pd.DataFrame(np.nan, index=list(CATEGORY_ORDER), columns=list(CATEGORY_ORDER))
    for label, value in part.items():
        a, b = label.split("-")
        matrix.loc[a, b] = matrix.loc[b, a] = value
    return matrix


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------


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


def group_size_curve(trials_by_participant: TrialsByParticipant, n_conds: int, *, sizes: tuple[int, ...] = GROUP_SIZES,
                     n_splits: int = N_SPLITS_GROUP, seed: int = SEED) -> pd.DataFrame:
    """Agreement between two disjoint random groups of each size (noise_ceiling.group_size_reliability, cached)."""
    def run() -> pd.DataFrame:
        return group_size_reliability(AGGREGATION, trials_by_participant=trials_by_participant, n_conds=n_conds,
                                      sizes=sizes, n_splits=n_splits, seed=seed)

    key = fingerprint(_digest(trials_by_participant), n_conds, sizes, n_splits, seed, AGGREGATION)
    return cached_frame(c.cache_path("group_size_curve.pkl"), key, run)


def ceiling_bracket(curve: pd.DataFrame, raw: pd.DataFrame) -> dict[str, float]:
    """The full-sample group ceiling on Spearman with a noise-free model, bracketed.

    `low` is sqrt of the empirical reliability at the largest group size measured (about half the sample),
    which understates the full sample's reliability; `high` is sqrt of the Spearman-Brown projection,
    which overstates it on these sparse RDMs.
    """
    largest = curve["group_size"].max()
    r_half = float(curve.loc[curve["group_size"] == largest, "spearman"].mean())
    r_sb = float(raw["spearman_brown"].mean())
    return {"group_size": int(largest), "r_half": r_half, "r_sb": r_sb,
            "low": float(np.sqrt(r_half)), "high": float(np.sqrt(r_sb))}


def nili_bounds(trials_by_participant: TrialsByParticipant, reference: np.ndarray, n_conds: int) -> pd.DataFrame:
    """Per participant: the fit to `reference`, the Nili lower bound (vs everyone else) and the Nili upper
    bound (vs everyone, self included), in tau-a and Spearman, and the participant's share of each group
    entry it rated. Cached."""
    def run() -> pd.DataFrame:
        lower = loo_model_fit(trials_by_participant, reference, n_conds, aggregation=AGGREGATION)
        upper = nili_upper_reliability(AGGREGATION, trials_by_participant=trials_by_participant, n_conds=n_conds)
        upper = upper.rename(columns={"fold_id": "participant_id", "kendall_a": "upper_kendall_a",
                                      "spearman": "upper_spearman"})
        return lower.merge(upper[["participant_id", "upper_kendall_a", "upper_spearman", "self_share"]],
                           on="participant_id")

    key = fingerprint(_digest(trials_by_participant), reference, n_conds, AGGREGATION, "nili")
    return cached_frame(c.cache_path("nili_bounds.pkl"), key, run)


def nili_summary(bounds: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across participants of the model fit and both bounds, per statistic."""
    rows = []
    for stat in ("kendall_a", "spearman"):
        for label, column in (("model fit (KM)", f"model_{stat}"), ("lower bound", f"lower_{stat}"),
                              ("upper bound", f"upper_{stat}")):
            values = bounds[column].dropna()
            rows.append({"statistic": c.STAT_LABEL[stat], "quantity": label,
                         "mean": float(values.mean()), "sd": float(values.std(ddof=1)), "n": len(values)})
    return pd.DataFrame(rows).set_index(["statistic", "quantity"])


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def joint_predictors() -> dict[str, np.ndarray]:
    """Predictor RDMs for behav_joint: the shared semantic RDMs, and pixels and CLIP as the mean of their
    pre- and post-SHINE RDMs (the pooled cohorts saw one variant each, about half and half)."""
    return {
        "sem_km": load_rdm("sem_km"), "sem_wn": load_rdm("sem_wn"),
        "clip": (load_rdm("clip_pre") + load_rdm("clip_post")) / 2.0,
        "pix": (load_rdm("sens_pre") + load_rdm("sens_post")) / 2.0,
    }


def _bootstrap_chunk(draws: list[tuple[int, np.random.SeedSequence]], trials: TrialsByParticipant,
                     predictors: dict[str, np.ndarray], n_conds: int, models: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    ids = list(trials)
    frames = []
    for b, seed_sequence in draws:
        drawn = np.random.default_rng(seed_sequence).choice(len(ids), size=len(ids), replace=True)
        rdm = _combine([t for i in drawn for t in trials[ids[i]]], n_conds, AGGREGATION)[0]
        frames.append(dec.fit_models(rdm, predictors, models=models).assign(boot=b))
    return pd.concat(frames, ignore_index=True)


def bootstrap_fits(
    trials: TrialsByParticipant, predictors: dict[str, np.ndarray], n_conds: int, *,
    models: dict[str, tuple[str, ...]] = JOINT_MODELS, n_boot: int = 1000, seed: int = SEED, n_jobs: int = 6,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Participant bootstrap of every model's fit to the joint RDM: each draw resamples the 155 participants
    with replacement, re-combines their trials once and refits every model. One row per (draw, model, term).
    Each draw has its own seed, so the result does not depend on `n_jobs`."""
    def run() -> pd.DataFrame:
        draws = list(enumerate(np.random.SeedSequence(seed).spawn(n_boot)))
        chunks = [draws[i::n_jobs * 4] for i in range(n_jobs * 4)] if n_jobs > 1 else [draws]
        frames = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_chunk)(chunk, trials, predictors, n_conds, models) for chunk in chunks if chunk
        )
        return pd.concat(frames, ignore_index=True).sort_values("boot", kind="stable", ignore_index=True)

    key = fingerprint(_digest(trials), *(predictors[t] for t in dec.TERMS), n_conds, AGGREGATION, list(models),
                      n_boot, seed)
    return cached_frame(cache_path, key, run)


def fit_estimates(observed: pd.DataFrame, boots: pd.DataFrame, stat: str = "beta", level: float = 0.95) -> pd.DataFrame:
    """Observed value of `stat` per (model, term) with its percentile bootstrap CI and the bootstrap mean.

    On these sparse RDMs a participant resample (about 63% distinct participants) behaves like a smaller,
    noisier group, so the draws are shifted below the full-sample estimate; `boot_mean` makes that visible.
    """
    rows = []
    for (model, term), draws in boots.groupby(["model", "term"], sort=False):
        lo, hi = percentile_ci(draws[stat].to_numpy(), level)
        value = observed.loc[(observed["model"] == model) & (observed["term"] == term), stat].iloc[0]
        rows.append({"model": model, "term": term, "estimate": float(value), "ci_low": lo, "ci_high": hi,
                     "boot_mean": float(draws[stat].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _behav_color(alpha: float | None = None) -> str:
    color = plotting.rdm_style(c.BEHAV).color
    return color if alpha is None else plotting._hex_to_rgba(color, alpha)


def fig_coverage(table: pd.DataFrame) -> go.Figure:
    """Fraction of pairs observed per pooled LCA bin and per stratum, with the overall fraction as a line."""
    overall = float(table.loc[("overall", "all"), "frac_observed"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=["per LCA bin", "per branch-depth stratum"],
                        shared_yaxes=True, horizontal_spacing=0.08)
    for col, grouping in enumerate(("LCA bin", "stratum"), start=1):
        part = table.loc[grouping]
        fig.add_trace(go.Bar(x=list(part.index), y=part["frac_observed"], marker_color=_behav_color(),
                             text=[f"{v:.1%}" for v in part["frac_observed"]], textposition="outside",
                             showlegend=False), row=1, col=col)
        fig.add_hline(y=overall, line={"color": _behav_color(), "dash": "dot", "width": 1}, row=1, col=col)
    fig.update_yaxes(title_text="fraction of pairs observed", range=[0, 1.08], row=1, col=1)
    fig.update_layout(title=f"Pair coverage of the joint behavioral RDM (dotted: overall {overall:.1%})",
                      template=_TEMPLATE, height=420, width=900)
    return fig


def fig_category_coverage(matrix: pd.DataFrame, *, title: str, fmt: str = ".1%") -> go.Figure:
    """Heatmap of a 6 x 6 category-pair matrix (category_matrix()), values printed in the cells."""
    labels = [f"{code}<br><sub>{CATEGORY_LABEL[code]}</sub>" for code in matrix.index]
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=labels, y=labels, colorscale="Purples", texttemplate=f"%{{z:{fmt}}}",
        colorbar={"tickformat": fmt},
    ))
    fig.update_layout(title=title, template=_TEMPLATE, height=520, width=620, yaxis={"autorange": "reversed"})
    return fig


def fig_distribution(values: pd.Series, *, x_title: str, title: str) -> go.Figure:
    """One per-participant distribution: histogram with each participant as a tick underneath."""
    fig = go.Figure(go.Histogram(x=values, nbinsx=30, marker_color=_behav_color(0.6), name="participants"))
    fig.add_trace(go.Scatter(x=values, y=np.zeros(len(values)), mode="markers", showlegend=False,
                             marker={"symbol": "line-ns-open", "size": 10, "color": _behav_color()}))
    fig.add_vline(x=float(values.median()), line={"color": plotting.OBSERVED_COLOR, "width": 2},
                  annotation_text=f"median {values.median():.3f}", annotation_position="top")
    fig.update_layout(title=title, template=_TEMPLATE, height=380, xaxis_title=x_title,
                      yaxis_title="participants", showlegend=False)
    return fig


def fig_nili(bounds: pd.DataFrame, stat: str = "kendall_a") -> go.Figure:
    """Per-participant model fit to KM next to the Nili lower and upper bounds (violins, points per participant)."""
    fig = go.Figure()
    for label, column, alpha in (("model fit (KM)", f"model_{stat}", 0.9), ("lower bound", f"lower_{stat}", 0.55),
                                 ("upper bound", f"upper_{stat}", 0.3)):
        fig.add_trace(go.Violin(y=bounds[column], name=label, box_visible=True, meanline_visible=True,
                                points="all", pointpos=0, jitter=0.4, line_color=_behav_color(),
                                fillcolor=_behav_color(alpha), marker={"color": _behav_color(), "size": 4}))
    fig.update_layout(title=f"Per-participant fit and Nili et al. (2014) bounds ({c.STAT_LABEL[stat]})",
                      template=_TEMPLATE, height=460, yaxis_title=c.STAT_LABEL[stat], showlegend=False)
    return fig


def fig_group_size(curve: pd.DataFrame, raw: pd.DataFrame, n_total: int) -> go.Figure:
    """Spearman between two disjoint groups against group size (mean +/- SD over draws), with the
    split-half value's Spearman-Brown projection to the full sample as a separate point."""
    summary = curve.groupby("group_size")["spearman"].agg(["mean", "std"])
    kwargs = plotting.rdm_scatter_kwargs(c.BEHAV)
    kwargs["name"] = "two disjoint groups (empirical)"
    fig = go.Figure(go.Scatter(x=summary.index, y=summary["mean"], mode="lines+markers",
                               error_y={"type": "data", "array": summary["std"]}, **kwargs))
    fig.add_trace(go.Scatter(x=[n_total], y=[float(raw["spearman_brown"].mean())], mode="markers",
                             name="Spearman-Brown projection of the split half",
                             marker={"color": plotting.NULL_COLOR, "symbol": "x", "size": 12}))
    fig.update_layout(title="Reliability of a group RDM by group size", template=_TEMPLATE, height=440, width=760,
                      xaxis_title="participants per group", yaxis_title="Spearman rho between groups",
                      legend={"orientation": "h", "y": -0.22})
    return fig


def fig_null_distances(distances: list[float], null: np.ndarray, *, n_bins: int = 80) -> go.Figure:
    """Pooled pairwise SpAM distances of the joint cohort against the random-placement null (overlaid density
    histograms). The histograms are binned here, so the figure stores bin heights rather than every distance."""
    edges = np.linspace(0.0, max(float(np.max(distances)), float(np.max(null))), n_bins + 1)
    centres, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    fig = go.Figure()
    for values, name, color in ((null, "random placement (null)", plotting.NULL_COLOR),
                                (distances, "participants (pooled)", _behav_color(0.6))):
        density, _ = np.histogram(values, bins=edges, density=True)
        fig.add_trace(go.Bar(x=centres, y=density, width=width, name=name, marker_color=color))
    fig.update_layout(barmode="overlay", template=_TEMPLATE, height=420, xaxis_title="normalised pairwise distance",
                      yaxis_title="density", title="Distances placed by participants vs random placement",
                      legend={"orientation": "h", "y": -0.2})
    return fig


def fig_betas(estimates: pd.DataFrame) -> go.Figure:
    """Forest plot of standardized NNLS betas (with bootstrap CIs) per model, each predictor in its family hue."""
    models = list(dict.fromkeys(estimates["model"]))
    fig = make_subplots(rows=len(models), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=[" + ".join(dec.TERM_LABEL[t] for t in dec.MODELS[m]) for m in models])
    for row, model in enumerate(models, start=1):
        part = estimates[estimates["model"] == model]
        for term, r in part.set_index("term").iterrows():
            color = dec._TERM_COLOR[term]
            fig.add_trace(go.Scatter(
                x=[r["estimate"]], y=[dec.TERM_LABEL[term]], mode="markers", showlegend=False,
                marker={"color": color, "size": 10, "symbol": plotting.RDM_VARIANT_SYMBOL["joint"]},
                error_x={"type": "data", "symmetric": False, "color": color, "thickness": 1.8, "width": 4,
                         "array": [r["ci_high"] - r["estimate"]], "arrayminus": [r["estimate"] - r["ci_low"]]},
            ), row=row, col=1)
        fig.add_vline(x=0, line={"color": "rgba(80,80,80,0.5)", "width": 1, "dash": "dot"}, row=row, col=1)
    fig.update_xaxes(title_text="standardized beta (NNLS)", row=len(models), col=1)
    fig.update_layout(template=_TEMPLATE, height=200 + 130 * len(models), width=760,
                      margin={"l": 150, "r": 30, "t": 70, "b": 60})
    return fig
