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

All statistics of a fit (NNLS betas and R^2, partial correlations, hierarchical delta R^2) are computed
from one correlation matrix of [perceptual, predictors], so fitting every model of MODELS on a bootstrap
draw costs one pass over the 262,450 pairs, not one per model.

Uncertainty is a participant bootstrap, as in reference_agreement.bootstrap_cohort_difference: each
cohort's participants are resampled with replacement, the combined RDMs rebuilt, and every model refit.

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
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from scipy.optimize import nnls

from analysis.s0_build_rdms.behavior.spam_ratings import _combine
from analysis.s0_build_rdms.common import load_rdm
from analysis.s1_pre_vs_post_shine.demographics_and_behavior import COHORTS, cohort_colors, cohort_label
from analysis.s1_pre_vs_post_shine.reference_agreement import TrialsByParticipant, _digest, percentile_ci
from analysis.utils.disk_cache import cached_frame, fingerprint
from analysis.utils.equivalence import equivalence_test_bootstrap_effect

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
# models with one semantic predictor, for which the pre-registered entry order (pixels, then CLIP,
# then semantic) defines a hierarchical delta R^2
HIERARCHICAL_MODELS = ("km", "km+clip", "wn", "wn+clip")

STATS = ("beta", "partial_r", "delta_r2", "r2")
STAT_LABEL = {"beta": "standardized beta", "partial_r": "partial correlation", "delta_r2": "delta R^2", "r2": "R^2"}
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


def _ols_r2(corr: np.ndarray, cols: list[int]) -> float:
    cov = corr[0, cols]
    return float(cov @ np.linalg.solve(corr[np.ix_(cols, cols)], cov))


def _partial_correlations(corr: np.ndarray, cols: list[int]) -> np.ndarray:
    """Correlation of y with each column, holding the other columns fixed."""
    precision = np.linalg.inv(corr[np.ix_([0, *cols], [0, *cols])])
    return -precision[0, 1:] / np.sqrt(precision[0, 0] * np.diag(precision)[1:])


def _delta_r2(corr: np.ndarray, terms: tuple[str, ...], column: dict[str, int]) -> list[float]:
    """Hierarchical delta R^2 per term: pixels first, then CLIP (if in the model), then the semantic term."""
    semantic = next(t for t in terms if t.startswith("sem"))
    entry = ["pix"] + (["clip"] if "clip" in terms else []) + [semantic]
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
    """One row per (model, term): NNLS beta, partial correlation, delta R^2 (hierarchical models only,
    NaN otherwise) and the model's NNLS R^2. Row order depends only on `models`."""
    corr = _correlation_matrix(perceptual, [predictors[t] for t in TERMS], mask)
    column = {t: i + 1 for i, t in enumerate(TERMS)}
    rows = []
    for name, terms in models.items():
        cols = [column[t] for t in terms]
        betas, r2 = _nnls(corr, cols)
        partial = _partial_correlations(corr, cols)
        delta = _delta_r2(corr, terms, column) if name in HIERARCHICAL_MODELS else [np.nan] * len(terms)
        rows += [
            {"model": name, "term": t, "beta": b, "partial_r": p, "delta_r2": d, "r2": r2}
            for t, b, p, d in zip(terms, betas, partial, delta)
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cohort comparison: observed fits and the participant bootstrap
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
    bootstrap draws are centred on. `pair_masks` as in bootstrap_decomposition()."""
    masks = {"own": None} if pair_masks is None else pair_masks
    frames = []
    for pairs, mask in masks.items():
        frame = _paired(fit_models(rdm_a, predictors_a, mask=mask), fit_models(rdm_b, predictors_b, mask=mask))
        frames.append(frame.assign(pairs=pairs))
    return pd.concat(frames, ignore_index=True)


def _bootstrap_draws(
    draws: list[tuple[int, np.random.SeedSequence]], trials_a: TrialsByParticipant, trials_b: TrialsByParticipant,
    predictors_a: dict[str, np.ndarray], predictors_b: dict[str, np.ndarray], n_conds: int, aggregation: str,
    masks: dict[str, np.ndarray | None],
) -> pd.DataFrame:
    """The bootstrap rows of the draws (draw index, its own seed) given, each fit on every pair set."""
    def combined_rdm(trials: TrialsByParticipant, rng: np.random.Generator) -> np.ndarray:
        ids = list(trials)
        drawn = rng.choice(len(ids), size=len(ids), replace=True)
        return _combine([t for i in drawn for t in trials[ids[i]]], n_conds, aggregation)[0]

    frames = []
    for b, seed_sequence in draws:
        rng = np.random.default_rng(seed_sequence)
        rdm_a, rdm_b = combined_rdm(trials_a, rng), combined_rdm(trials_b, rng)
        for pairs, mask in masks.items():
            paired = _paired(fit_models(rdm_a, predictors_a, mask=mask), fit_models(rdm_b, predictors_b, mask=mask))
            frames.append(paired.assign(boot=b, pairs=pairs))
    return pd.concat(frames, ignore_index=True)


def bootstrap_decomposition(
    trials_a: TrialsByParticipant, trials_b: TrialsByParticipant,
    predictors_a: dict[str, np.ndarray], predictors_b: dict[str, np.ndarray], n_conds: int,
    *, aggregation: str = "simple", n_boot: int = 1000, seed: int = 0,
    pair_masks: dict[str, np.ndarray | None] | None = None, n_jobs: int = 1, cache_path: Path | None = None,
) -> pd.DataFrame:
    """Participant bootstrap of every model's fit in both cohorts and of their A - B difference.

    The cohorts are disjoint participants, so each is resampled on its own; per draw both are
    re-combined with `aggregation` once and every model of MODELS is refit on each pair set.
    `pair_masks` names the pair sets (default {"own": None}, each cohort on the pairs it observed); a
    boolean mask over the condensed pairs restricts both cohorts to those pairs. One row per (draw,
    pair set, model, term) with `<stat>_a`, `<stat>_b`, `<stat>_diff` for each of STATS.

    Each draw has its own seed (spawned from `seed`), so the result does not depend on `n_jobs`, the
    number of worker processes the draws are spread over (the combination step dominates the cost).
    """
    masks = {"own": None} if pair_masks is None else pair_masks

    def run() -> pd.DataFrame:
        draws = list(enumerate(np.random.SeedSequence(seed).spawn(n_boot)))
        chunks = [draws[i::n_jobs * 4] for i in range(n_jobs * 4)] if n_jobs > 1 else [draws]
        frames = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_draws)(chunk, trials_a, trials_b, predictors_a, predictors_b, n_conds, aggregation, masks)
            for chunk in chunks if chunk
        )
        return pd.concat(frames, ignore_index=True).sort_values("boot", kind="stable", ignore_index=True)

    key = fingerprint(
        _digest(trials_a), _digest(trials_b), *(predictors_a[t] for t in TERMS), *(predictors_b[t] for t in TERMS),
        n_conds, aggregation, n_boot, seed, list(masks), *masks.values(),
    )
    return cached_frame(cache_path, key, run)


def cohort_estimates(observed: pd.DataFrame, boots: pd.DataFrame, stat: str = "beta", level: float = 0.95) -> pd.DataFrame:
    """Per cohort, pair set, model and term: the observed `stat` with its percentile bootstrap CI.
    Columns pairs, model, term, cohort, estimate, ci_low, ci_high (cohort in COHORTS)."""
    rows = []
    for (pairs, model, term), draws in boots.groupby(["pairs", "model", "term"], sort=False):
        point = observed[(observed["pairs"] == pairs) & (observed["model"] == model) & (observed["term"] == term)].iloc[0]
        for cohort, suffix in zip(COHORTS, ("a", "b")):
            low, high = percentile_ci(draws[f"{stat}_{suffix}"].to_numpy(), level)
            rows.append({"pairs": pairs, "model": model, "term": term, "cohort": cohort,
                         "estimate": point[f"{stat}_{suffix}"], "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def difference_table(
    observed: pd.DataFrame, boots: pd.DataFrame, n_a: int, n_b: int, stat: str = "beta",
) -> pd.DataFrame:
    """One row per (pair set, model, term) on D = A - B (here pre - post): the observed effect, its
    95% percentile CI, `a_exceeds_b` (95% CI entirely above 0, the H3(i) criterion) and the TOST
    equivalence result (equivalence_test_bootstrap_effect: 90% CI inside the 80%-power bound)."""
    rows = []
    for (pairs, model, term), draws in boots.groupby(["pairs", "model", "term"], sort=False):
        point = observed[(observed["pairs"] == pairs) & (observed["model"] == model) & (observed["term"] == term)].iloc[0]
        effect = point[f"{stat}_diff"]
        tost = equivalence_test_bootstrap_effect(
            effect, draws[f"{stat}_diff"], n_a=n_a, n_b=n_b, label=f"pre - post, {model}, {TERM_LABEL[term]}",
        )
        low95, high95 = percentile_ci(draws[f"{stat}_diff"].to_numpy(), 0.95)
        rows.append({"pairs": pairs, "model": model, "term": term, "stat": stat, **tost,
                     "ci95_low": low95, "ci95_high": high95, "a_exceeds_b": bool(low95 > 0)})
    return pd.DataFrame(rows)


def zero_fraction(boots: pd.DataFrame) -> pd.DataFrame:
    """Share of bootstrap draws in which each NNLS beta is exactly 0 (the constraint is active), per
    cohort; columns pairs, model, term, a, b."""
    flags = boots.assign(a=boots["beta_a"] == 0.0, b=boots["beta_b"] == 0.0)
    return flags.groupby(["pairs", "model", "term"], sort=False)[["a", "b"]].mean().reset_index()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_estimates(estimates: pd.DataFrame, models: list[str], *, y_title: str) -> go.Figure:
    """Pre and post estimate of every term with its CI, one panel per model (cohort_estimates() output)."""
    fig = make_subplots(rows=1, cols=len(models), subplot_titles=models, shared_yaxes=True, horizontal_spacing=0.04)
    for col, model in enumerate(models, start=1):
        for cohort in COHORTS:
            rows = estimates[(estimates["model"] == model) & (estimates["cohort"] == cohort)]
            color = cohort_colors(cohort)["violin_line"]
            fig.add_trace(go.Scatter(
                x=[TERM_LABEL[t] for t in rows["term"]], y=rows["estimate"], mode="markers", name=cohort_label(cohort),
                marker={"color": color, "size": 10}, showlegend=col == 1, legendgroup=cohort,
                error_y={"type": "data", "symmetric": False, "color": color,
                         "array": rows["ci_high"] - rows["estimate"], "arrayminus": rows["estimate"] - rows["ci_low"]},
            ), row=1, col=col)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_layout(template=_TEMPLATE, height=440, width=max(520, 330 * len(models)),
                      legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"})
    return fig


def fig_delta_r2(observed: pd.DataFrame, models: list[str] = list(HIERARCHICAL_MODELS)) -> go.Figure:
    """Stacked hierarchical delta R^2 (pixels, then CLIP, then semantic) per cohort and model."""
    fig = go.Figure()
    for term in ("pix", "clip", "sem_km", "sem_wn"):
        for cohort, suffix in zip(COHORTS, ("a", "b")):
            rows = observed[(observed["term"] == term) & observed["model"].isin(models)]
            if rows.empty:
                continue
            fig.add_trace(go.Bar(
                x=[[m for m in rows["model"]], [cohort_label(cohort)] * len(rows)], y=rows[f"delta_r2_{suffix}"],
                name=TERM_LABEL[term], legendgroup=term, showlegend=cohort == COHORTS[0],
            ))
    fig.update_layout(template=_TEMPLATE, barmode="relative", height=440, yaxis_title="delta R^2 (hierarchical)")
    return fig
