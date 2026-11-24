"""
Semantic + sensory decomposition of each cohort's perceptual RDM, and its pre vs post comparison
(analysis/s1_pre_vs_post_shine/6_decomposition.ipynb): pre-registered H3,
`D_perc = a*D_sem + b*D_CLIP + g*D_pix + e`, fit per cohort by non-negative least squares.

Every variable (the perceptual RDM and each predictor) is z-scored over the pairs of the fit, so the
coefficients are standardized betas, comparable across predictors and across cohorts, and the intercept
is 0 by construction. The perceptual RDM is only partly observed (NaN = unrated), so each fit uses the
pairs the cohort observed (optionally intersected with a `mask`, e.g. the pairs both cohorts observed).
The predictors are variant-specific where the stimuli are (pixels and CLIP of the cohort's own SHINE
variant) and shared where they are not (the KM and WordNet semantic RDMs).

All statistics of a fit (NNLS betas and R^2, unconstrained OLS betas as a sensitivity check, partial
correlations, hierarchical delta R^2) are computed
from one correlation matrix of [perceptual, predictors], so fitting every model of MODELS on a resampled
RDM costs one pass over the 262,450 pairs, not one per model.

Uncertainty is a delete-one participant jackknife, as in reference_agreement.jackknife_cohort_difference:
each cohort's participants are left out in turn, that cohort's combined RDM rebuilt, and every model refit;
CIs are centred on the full-sample estimate. A participant bootstrap was used before; on these sparse RDMs
its draws are attenuated (a resample has about 63% distinct participants), so its CIs sat below the
estimates (see analysis.s0_build_rdms.behavior.resampling).

Design decision: this stays one notebook with this one flat module. Reconsider splitting it into a
confirmatory and a robustness notebook only if the pre-registered exploratory variants (4-predictor VGG
FC7, SGPT semantic source) are added.

Usage (from repo root):
    from analysis.s1_pre_vs_post_shine import decomposition as dec
    fits = dec.fit_models(perceptual_rdm, dec.predictors_for("pre"))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls
from scipy.stats import norm

from analysis.s0_build_rdms.behavior.resampling import participant_jackknife
from analysis.s0_build_rdms.behavior.spam_ratings import _combine
from analysis.s0_build_rdms.common import load_rdm
from analysis.s1_pre_vs_post_shine.demographics_and_behavior import COHORTS, cohort_colors, cohort_label, cohort_symbol
from analysis.s1_pre_vs_post_shine.reference_agreement import TrialsByParticipant, _digest, difference_se
from analysis.utils.disk_cache import cached_frame, fingerprint
from analysis.utils.equivalence import equivalence_test_se_effect
from analysis.utils.plotting import RDM_FAMILY_HUE, add_verdict_legend, verdict_color
from analysis.utils.stats import jackknife_bias, jackknife_ci

TERMS = ("sem_km", "sem_wn", "clip", "pix")
TERM_LABEL = {"sem_km": "semantic (KM)", "sem_wn": "semantic (WordNet)", "clip": "CLIP", "pix": "pixels"}

# name -> predictors entered (the confirmatory H3 model is "km+clip")
MODELS: dict[str, tuple[str, ...]] = {
    "km": ("sem_km", "pix"),
    "km+clip": ("sem_km", "clip", "pix"),
    "wn": ("sem_wn", "pix"),
    "wn+clip": ("sem_wn", "clip", "pix"),
    "km+wn": ("sem_km", "sem_wn", "pix"),
    "km+wn+clip": ("sem_km", "sem_wn", "clip", "pix"),
}

STATS = ("beta", "beta_ols", "partial_r", "delta_r2", "r2")
STAT_LABEL = {
    "beta": "standardized beta (NNLS)", "beta_ols": "standardized beta (OLS)", "partial_r": "partial correlation",
    "delta_r2": "delta R^2", "r2": "R^2",
}
_RIDGE = 1e-12
_TEMPLATE = "plotly_white"


def predictors_for(cohort: str) -> dict[str, np.ndarray]:
    """Condensed predictor RDMs for one cohort: shared semantic RDMs, that cohort's own pixels and CLIP."""
    return {
        "sem_km": load_rdm("sem_km"), "sem_wn": load_rdm("sem_wn"),
        "clip": load_rdm(f"clip_{cohort}"), "pix": load_rdm(f"sens_{cohort}"),
    }


# ---------------------------------------------------------------------------
# One fit: everything from a single correlation matrix
# ---------------------------------------------------------------------------


def _correlation_matrix(y: np.ndarray, columns: list[np.ndarray], mask: np.ndarray | None) -> np.ndarray:
    """Pearson correlations of [y, *columns] over the pairs where y is observed (and inside `mask`)."""
    keep = np.isfinite(y)
    if mask is not None:
        keep &= mask
    return np.corrcoef(np.column_stack([y[keep], *(c[keep] for c in columns)]), rowvar=False)


def _nnls(corr: np.ndarray, cols: list[int]) -> tuple[np.ndarray, float]:
    """Non-negative standardized betas and R^2 of y on the columns `cols` of a correlation matrix.

    min ||y - Xb||^2 over b >= 0 is min b'Gb - 2c'b with G the predictors' and c their correlations
    with y (all unit variance), which NNLS solves on the Cholesky factor of G."""
    gram, cov = corr[np.ix_(cols, cols)], corr[0, cols]
    factor = np.linalg.cholesky(gram + _RIDGE * np.eye(len(cols)))
    betas, _ = nnls(factor.T, np.linalg.solve(factor, cov))
    return betas, float(2.0 * cov @ betas - betas @ gram @ betas)


def _ols(corr: np.ndarray, cols: list[int]) -> tuple[np.ndarray, float]:
    """Unconstrained standardized betas and R^2 of y on the columns `cols` of a correlation matrix."""
    cov = corr[0, cols]
    betas = np.linalg.solve(corr[np.ix_(cols, cols)], cov)
    return betas, float(cov @ betas)


def _ols_r2(corr: np.ndarray, cols: list[int]) -> float:
    return _ols(corr, cols)[1]


def _partial_correlations(corr: np.ndarray, cols: list[int]) -> np.ndarray:
    """Correlation of y with each column, holding the other columns fixed."""
    precision = np.linalg.inv(corr[np.ix_([0, *cols], [0, *cols])])
    return -precision[0, 1:] / np.sqrt(precision[0, 0] * np.diag(precision)[1:])


def _delta_r2(corr: np.ndarray, terms: tuple[str, ...], column: dict[str, int]) -> list[float]:
    """Hierarchical delta R^2 per term: pixels first, then CLIP (if in the model), then the semantic terms
    in model order (so in km+wn, WordNet's share is what it adds beyond KM)."""
    entry = ["pix"] + (["clip"] if "clip" in terms else []) + [t for t in terms if t.startswith("sem")]
    r2_so_far = [_ols_r2(corr, [column[t] for t in entry[: i + 1]]) for i in range(len(entry))]
    gain = {t: r2 - previous for t, r2, previous in zip(entry, r2_so_far, [0.0, *r2_so_far])}
    return [gain[t] for t in terms]


def standardized_nnls(y: np.ndarray, X: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Standardized non-negative betas and R^2 of y on the columns of X, over the pairs finite in y
    (and inside `mask`). X is (n_pairs, k) and must be fully observed."""
    corr = _correlation_matrix(y, [X[:, j] for j in range(X.shape[1])], mask)
    return _nnls(corr, list(range(1, X.shape[1] + 1)))


def fit_models(
    perceptual: np.ndarray, predictors: dict[str, np.ndarray], *, mask: np.ndarray | None = None,
    models: dict[str, tuple[str, ...]] = MODELS,
) -> pd.DataFrame:
    """One row per (model, term): NNLS beta, OLS beta, partial correlation, hierarchical delta R^2 and the
    model's NNLS R^2. Row order depends only on `models`."""
    corr = _correlation_matrix(perceptual, [predictors[t] for t in TERMS], mask)
    column = {t: i + 1 for i, t in enumerate(TERMS)}
    rows = []
    for name, terms in models.items():
        cols = [column[t] for t in terms]
        betas, r2 = _nnls(corr, cols)
        ols_betas, _ = _ols(corr, cols)
        partial = _partial_correlations(corr, cols)
        delta = _delta_r2(corr, terms, column)
        rows += [
            {"model": name, "term": t, "beta": b, "beta_ols": o, "partial_r": p, "delta_r2": d, "r2": r2}
            for t, b, o, p, d in zip(terms, betas, ols_betas, partial, delta)
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cohort comparison: observed fits and the participant jackknife
# ---------------------------------------------------------------------------


def _paired(fit_a: pd.DataFrame, fit_b: pd.DataFrame) -> pd.DataFrame:
    """Cohort A and B fits of identical row order side by side, with the A - B difference of each statistic."""
    out = fit_a[["model", "term"]].copy()
    for stat in STATS:
        out[f"{stat}_a"], out[f"{stat}_b"] = fit_a[stat].to_numpy(), fit_b[stat].to_numpy()
        out[f"{stat}_diff"] = out[f"{stat}_a"] - out[f"{stat}_b"]
    return out


def observed_decomposition(
    rdm_a: np.ndarray, rdm_b: np.ndarray, predictors_a: dict[str, np.ndarray], predictors_b: dict[str, np.ndarray],
    *, pair_masks: dict[str, np.ndarray | None] | None = None,
) -> pd.DataFrame:
    """Full-data fits of both cohorts, one row per (pair set, model, term): the point estimates the
    jackknife CIs are centred on. `pair_masks` as in jackknife_decomposition()."""
    masks = {"own": None} if pair_masks is None else pair_masks
    frames = []
    for pairs, mask in masks.items():
        frame = _paired(fit_models(rdm_a, predictors_a, mask=mask), fit_models(rdm_b, predictors_b, mask=mask))
        frames.append(frame.assign(pairs=pairs))
    return pd.concat(frames, ignore_index=True)


def _jackknife_score(side: str, other: np.ndarray, predictors_a: dict[str, np.ndarray],
                     predictors_b: dict[str, np.ndarray], masks: dict[str, np.ndarray | None]):
    """Scorer for one cohort's leave-one-out RDMs, paired with the other cohort's full RDM, on every pair set."""
    def score(rdm: np.ndarray) -> list[dict]:
        rdm_a, rdm_b = (rdm, other) if side == "a" else (other, rdm)
        rows = []
        for pairs, mask in masks.items():
            paired = _paired(fit_models(rdm_a, predictors_a, mask=mask), fit_models(rdm_b, predictors_b, mask=mask))
            rows += paired.assign(side=side, pairs=pairs).to_dict("records")
        return rows

    return score


def jackknife_decomposition(
    trials_a: TrialsByParticipant, trials_b: TrialsByParticipant,
    predictors_a: dict[str, np.ndarray], predictors_b: dict[str, np.ndarray], n_conds: int,
    *, aggregation: str = "simple", pair_masks: dict[str, np.ndarray | None] | None = None, n_jobs: int = 6,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Delete-one participant jackknife of every model's fit in both cohorts and of their A - B difference.

    The cohorts are disjoint participants, so each is jackknifed on its own with the other at full size:
    each participant is left out in turn, that cohort's RDM re-combined with `aggregation` and every model of
    MODELS refit on each pair set. `pair_masks` names the pair sets (default {"own": None}, each cohort on
    the pairs it observed); a boolean mask over the condensed pairs restricts both cohorts to those pairs.
    One row per (side, left-out participant, pair set, model, term), `side` naming the cohort that lost a
    participant, with `<stat>_a`, `<stat>_b`, `<stat>_diff` for each of STATS. SEs: jackknife_se over the
    rows of one side (a cohort's own statistic) or reference_agreement.difference_se (the difference).
    """
    masks = {"own": None} if pair_masks is None else pair_masks

    def run() -> pd.DataFrame:
        full = {side: _combine([t for ts in trials.values() for t in ts], n_conds, aggregation)[0]
                for side, trials in (("a", trials_a), ("b", trials_b))}
        frames = [
            participant_jackknife(trials, n_conds,
                                  _jackknife_score(side, full["b" if side == "a" else "a"], predictors_a, predictors_b, masks),
                                  aggregation=aggregation, n_jobs=n_jobs)
            for side, trials in (("a", trials_a), ("b", trials_b))
        ]
        return pd.concat(frames, ignore_index=True)

    key = fingerprint(
        _digest(trials_a), _digest(trials_b), *(predictors_a[t] for t in TERMS), *(predictors_b[t] for t in TERMS),
        n_conds, aggregation, list(masks), *masks.values(), "jackknife",
    )
    return cached_frame(cache_path, key, run)


def cohort_estimates(observed: pd.DataFrame, jack: pd.DataFrame, stat: str = "beta", level: float = 0.95) -> pd.DataFrame:
    """Per cohort, pair set, model and term: the observed `stat` with its jackknife CI (centred on it) and
    the jackknife bias estimate (a diagnostic of how much the statistic grows with the number of participants).
    Columns pairs, model, term, cohort, estimate, ci_low, ci_high, bias (cohort in COHORTS)."""
    rows = []
    for (pairs, model, term), draws in jack.groupby(["pairs", "model", "term"], sort=False):
        if draws[f"{stat}_a"].isna().all():
            continue
        point = observed[(observed["pairs"] == pairs) & (observed["model"] == model) & (observed["term"] == term)].iloc[0]
        for cohort, suffix in zip(COHORTS, ("a", "b")):
            values = draws.loc[draws["side"] == suffix, f"{stat}_{suffix}"].to_numpy()
            low, high = jackknife_ci(point[f"{stat}_{suffix}"], values, level)
            rows.append({"pairs": pairs, "model": model, "term": term, "cohort": cohort,
                         "estimate": point[f"{stat}_{suffix}"], "ci_low": low, "ci_high": high,
                         "bias": jackknife_bias(point[f"{stat}_{suffix}"], values)})
    return pd.DataFrame(rows)


def difference_table(
    observed: pd.DataFrame, jack: pd.DataFrame, n_a: int, n_b: int, stat: str = "beta",
) -> pd.DataFrame:
    """One row per (pair set, model, term) on D = A - B (here pre - post): the observed effect, its 95% CI
    (D +/- z * SE, SE the jackknife SE of the difference), `a_exceeds_b` (95% CI entirely above 0, the H3(i)
    criterion) and the TOST equivalence result (equivalence_test_se_effect: 90% CI inside the 80%-power bound)."""
    rows = []
    z95 = float(norm.ppf(0.975))
    for (pairs, model, term), draws in jack.groupby(["pairs", "model", "term"], sort=False):
        if draws[f"{stat}_diff"].isna().all():
            continue
        point = observed[(observed["pairs"] == pairs) & (observed["model"] == model) & (observed["term"] == term)].iloc[0]
        effect = point[f"{stat}_diff"]
        se = difference_se(draws, f"{stat}_diff")
        tost = equivalence_test_se_effect(effect, se, n_a=n_a, n_b=n_b, label=f"pre - post, {model}, {TERM_LABEL[term]}")
        low95, high95 = effect - z95 * se, effect + z95 * se
        rows.append({"pairs": pairs, "model": model, "term": term, "stat": stat, **tost,
                     "ci95_low": low95, "ci95_high": high95, "a_exceeds_b": bool(low95 > 0)})
    return pd.DataFrame(rows)


def model_summary(estimates: pd.DataFrame, differences: pd.DataFrame, model: str, *, pairs: str = "own") -> pd.DataFrame:
    """One row per term of `model`: each cohort's estimate with its 95% CI, the pre - post difference with
    its 95% CI, whether that CI lies above 0 and whether the TOST equivalence test passed."""
    def interval(row: pd.Series) -> str:
        return f"{row['estimate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"

    est = estimates[(estimates["model"] == model) & (estimates["pairs"] == pairs)]
    diff = differences[(differences["model"] == model) & (differences["pairs"] == pairs)].set_index("term")
    rows = []
    for term, group in est.groupby("term", sort=False):
        by_cohort = {r["cohort"]: interval(r) for _, r in group.iterrows()}
        d = diff.loc[term]
        rows.append({
            "term": TERM_LABEL[term], **by_cohort,
            "pre - post": f"{d['effect']:+.3f} [{d['ci95_low']:+.3f}, {d['ci95_high']:+.3f}]",
            "pre > post": d["a_exceeds_b"], "equivalent": d["equivalent"], "bound": d["bound"],
        })
    return pd.DataFrame(rows).set_index("term")


def zero_fraction(jack: pd.DataFrame) -> pd.DataFrame:
    """Share of each cohort's leave-one-out refits in which each NNLS beta is exactly 0 (the constraint is
    active); columns pairs, model, term, a, b."""
    shares = {}
    for side in ("a", "b"):
        own = jack[jack["side"] == side]
        shares[side] = (own["beta_" + side] == 0.0).groupby([own["pairs"], own["model"], own["term"]], sort=False).mean()
    return pd.DataFrame(shares).reset_index()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


_TERM_COLOR = {term: RDM_FAMILY_HUE[family]  # each predictor RDM in its family hue (shared RDM design language)
               for term, family in {"pix": "sens", "clip": "clip", "sem_km": "km", "sem_wn": "wn"}.items()}
_DODGE = 0.17  # vertical offset of the pre and post marker within a term's row
_ROW_PX, _GAP_PX = 42, 95  # forest panels: height per term, space between panels (title and tick labels)
_SEMANTIC_NAME = {"sem_km": "KM", "sem_wn": "WordNet"}
DELTA_R2_PANELS = (("km", "km+clip"), ("wn", "wn+clip"), ("km+wn", "km+wn+clip"))


def _model_title(model: str) -> str:
    return " + ".join(TERM_LABEL[t] for t in MODELS[model])


def _stacked_forest(models: list[str], x_title: str) -> go.Figure:
    """One panel per model stacked vertically on a shared x axis, each as tall as the model has terms, the
    terms on the y axis (first on top). Tick labels are kept on every panel and the panels are spaced to
    leave room for them and for the panel title, which is left-aligned above the panel."""
    n_terms = [len(MODELS[m]) for m in models]
    heights = [40 + _ROW_PX * n for n in n_terms]
    plot_px = sum(heights) + _GAP_PX * (len(models) - 1)
    fig = make_subplots(
        rows=len(models), cols=1, row_heights=heights, vertical_spacing=_GAP_PX / plot_px,
        subplot_titles=[_model_title(m) for m in models],
    )
    for row, (model, n) in enumerate(zip(models, n_terms), start=1):
        fig.update_yaxes(
            tickmode="array", tickvals=list(range(n)), ticktext=[TERM_LABEL[t] for t in MODELS[model]],
            range=[n - 0.5, -0.5], showgrid=False, zeroline=False, row=row, col=1,
        )
        fig.update_xaxes(matches="x", showticklabels=True, row=row, col=1)
        fig.add_vline(x=0, line={"color": "rgba(80,80,80,0.5)", "width": 1, "dash": "dot"}, row=row, col=1)
    fig.update_xaxes(title_text=x_title, row=len(models), col=1)
    for annotation in fig.layout.annotations:
        annotation.update(x=0, xanchor="left", font={"size": 13})
    fig.update_layout(
        template=_TEMPLATE, height=plot_px + 150, margin={"l": 150, "r": 30, "t": 70, "b": 60},
        legend={"orientation": "h", "x": 0, "y": 1, "yanchor": "top", "yref": "container"},
    )
    return fig


def fig_estimates(estimates: pd.DataFrame, models: list[str], *, x_title: str) -> go.Figure:
    """Forest plot of every model's weights (cohort_estimates() output): predictors on the y axis, pre and
    post estimate with its CI on a shared x axis, one panel per model."""
    fig = _stacked_forest(models, x_title)
    for row, model in enumerate(models, start=1):
        terms = list(MODELS[model])
        for cohort, side in zip(COHORTS, (-1, 1)):
            rows = estimates[(estimates["model"] == model) & (estimates["cohort"] == cohort)].set_index("term").loc[terms]
            color = cohort_colors(cohort)["violin_line"]
            fig.add_trace(go.Scatter(
                x=rows["estimate"], y=[i + side * _DODGE for i in range(len(terms))], mode="markers",
                name=cohort_label(cohort), marker={"color": color, "size": 9, "symbol": cohort_symbol(cohort)},
                showlegend=row == 1, legendgroup=cohort,
                error_x={"type": "data", "symmetric": False, "color": color, "thickness": 1.8, "width": 4,
                         "array": rows["ci_high"] - rows["estimate"], "arrayminus": rows["estimate"] - rows["ci_low"]},
                customdata=np.column_stack([rows["ci_low"], rows["ci_high"]]),
                hovertemplate=f"{cohort_label(cohort)}<br>%{{x:.3f}} [%{{customdata[0]:.3f}}, %{{customdata[1]:.3f}}]<extra></extra>",
            ), row=row, col=1)
    return fig


def fig_difference_forest(differences: pd.DataFrame, models: list[str], *, x_title: str) -> go.Figure:
    """Forest plot of the pre - post difference of every weight (difference_table() output, one pair set and
    statistic): the effect with its 90% TOST interval and the +/- equivalence bound as two ticks on the
    same row; blue rows are equivalent to 0, red ones are not (the shared verdict colours). One panel per model, shared x axis."""
    fig = _stacked_forest(models, x_title)
    for row, model in enumerate(models, start=1):
        rows = differences[differences["model"] == model].set_index("term")
        for i, term in enumerate(MODELS[model]):
            r = rows.loc[term]
            color = verdict_color(r["equivalent"])
            fig.add_trace(go.Scatter(
                x=[r["effect"]], y=[i], mode="markers", marker={"color": color, "size": 9}, showlegend=False,
                error_x={"type": "data", "symmetric": False, "color": color, "thickness": 1.8, "width": 4,
                         "array": [r["ci_high"] - r["effect"]], "arrayminus": [r["effect"] - r["ci_low"]]},
                hovertemplate=f"{TERM_LABEL[term]}<br>effect=%{{x:.3f}}, 90% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]<extra></extra>",
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=[-r["bound"], r["bound"]], y=[i, i], mode="markers", showlegend=False, hoverinfo="skip",
                marker={"color": "rgba(80,80,80,0.8)", "size": 14, "symbol": "line-ns-open", "line": {"width": 2}},
            ), row=row, col=1)
    add_verdict_legend(fig, row=1, col=1)
    return fig


def fig_delta_r2(observed: pd.DataFrame, panels: tuple[tuple[str, ...], ...] = DELTA_R2_PANELS) -> go.Figure:
    """Stacked hierarchical delta R^2 (pixels, then CLIP, then the semantic predictors) per model and cohort,
    one panel per tuple of `panels` models, on a shared y axis and one shared legend (one colour per predictor,
    the same for both cohorts)."""
    fig = make_subplots(
        rows=1, cols=len(panels), shared_yaxes=True, horizontal_spacing=0.07,
        subplot_titles=[" + ".join(_SEMANTIC_NAME[t] for t in MODELS[p[0]] if t.startswith("sem")) for p in panels],
    )
    shown: set[str] = set()
    for col, models in enumerate(panels, start=1):
        for term in ("pix", "clip", "sem_km", "sem_wn"):
            for cohort, suffix in zip(COHORTS, ("a", "b")):
                rows = observed[(observed["term"] == term) & observed["model"].isin(models)]
                if rows.empty:
                    continue
                fig.add_trace(go.Bar(
                    x=[list(rows["model"]), [cohort_label(cohort).split("-")[0]] * len(rows)], y=rows[f"delta_r2_{suffix}"],
                    name=TERM_LABEL[term], legendgroup=term, showlegend=term not in shown, marker={"color": _TERM_COLOR[term]},
                ), row=1, col=col)
                shown.add(term)
    fig.update_yaxes(title_text="delta R^2 (hierarchical)", row=1, col=1)
    fig.update_xaxes(tickangle=0)
    fig.update_layout(
        template=_TEMPLATE, barmode="relative", height=470, margin={"l": 70, "r": 20, "t": 110, "b": 70},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.2, "yanchor": "bottom"},
    )
    return fig
