"""
Pre- vs post-SHINE cohort comparison, structure layer (analysis/s1_pre_vs_post_shine/
3_geometry.ipynb): MDS-based structural comparison of the two cohorts'
behavioral RDMs (weighted-SMACOF stress, embedding eigenvalue spectra, cross-cohort
agreement), within- vs between-cohort RDM agreement at a matched group size, the
cohort-label permutation test of pre-vs-post RDM agreement, and
its equivalence counterpart (delete-a-group jackknife CI of observed minus the
exchangeable-null expectation).

MDS-embedding logic itself (embedding eigenspectrum, cached SMACOF sweep,
cross-embedding comparison) lives in analysis/utils/mds.py.

Usage (from repo root):
    from analysis.s1_pre_vs_post_shine.geometry import (
        MDS_CACHE_DIR, PERMUTATION_CACHE_DIR, fig_stress_scree, fig_embedding_spectrum,
        fig_cross_cohort_agreement, within_between_agreement, fig_within_between,
        permutation_test_cohort_agreement, fig_permutation_null, equivalence_test_cohort_agreement,
    )
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from analysis.s1_pre_vs_post_shine.demographics_and_behavior import COHORTS, cohort_colors, cohort_label, cohort_symbol
from analysis.s0_build_rdms.behavior.noise_ceiling import _parse_trials_by_participant
from analysis.s0_build_rdms.behavior.spam_ratings import _combine
from analysis.s0_build_rdms.common import REPO_ROOT, image_manifest_index
from analysis.utils.disk_cache import cached_array
from analysis.utils.equivalence import resolvable_bound, tost_bound_multiplier
from analysis.utils.plotting import NULL_COLOR, OBSERVED_COLOR, stat_scatter_kwargs
from analysis.utils.stats import masked_rank_correlation

RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "pre_vs_post_shine"
MDS_CACHE_DIR = RESULTS_DIR / "mds_cache"
PERMUTATION_CACHE_DIR = RESULTS_DIR / "permutation_cache"

RdmBuilder = Callable[[Sequence[str]], np.ndarray]

_worker_builder: RdmBuilder | None = None


def fig_embedding_spectrum(spectra_by_dim: dict[int, dict[str, np.ndarray]]) -> go.Figure:
    """One subplot per embedding dimensionality (shared y-axis), each with a
    line+marker eigenvalue spectrum per cohort: the share of variance along
    each principal axis of the fitted SMACOF embedding
    (analysis.utils.mds.embedding_eigenspectrum(), normalised)."""
    dims = list(spectra_by_dim)
    fig = make_subplots(
        rows=1, cols=len(dims), shared_yaxes=True, subplot_titles=[f"{d} dimensions" for d in dims],
        horizontal_spacing=0.04,
    )
    for col_idx, dim in enumerate(dims, start=1):
        for cohort in COHORTS:
            cc = cohort_colors(cohort)
            fig.add_trace(go.Scatter(
                x=list(range(1, dim + 1)), y=spectra_by_dim[dim][cohort].tolist(),
                mode="lines+markers", line={"color": cc["main"], "width": 2.5},
                marker={"size": 7, "color": cc["main"], "symbol": cohort_symbol(cohort)}, name=cohort_label(cohort), legendgroup=cohort,
                showlegend=col_idx == 1,
            ), row=1, col=col_idx)
        fig.update_xaxes(title_text="Component", tickvals=list(range(1, dim + 1)), row=1, col=col_idx)
    fig.update_yaxes(title_text="Proportion of variance (eigenvalue)", row=1, col=1)
    fig.update_layout(
        template="plotly_white", title="Eigenvalue spectrum of the MDS embeddings",
        height=420, margin={"l": 70, "r": 40, "t": 80, "b": 60},
    )
    return fig


def fig_stress_scree(mds_results_by_cohort: dict[str, dict[int, dict]], dims: list[int]) -> go.Figure:
    """Line+marker scree plot of weighted-SMACOF stress across embedding
    dimensionality, one trace per cohort."""
    fig = go.Figure()
    for cohort in COHORTS:
        cc = cohort_colors(cohort)
        stresses = [mds_results_by_cohort[cohort][d]["stress"] for d in dims]
        fig.add_trace(go.Scatter(
            x=dims, y=stresses, mode="lines+markers",
            line={"color": cc["main"], "width": 2.5}, marker={"size": 8, "color": cc["main"], "symbol": cohort_symbol(cohort)},
            name=cohort_label(cohort),
        ))
    fig.update_xaxes(title_text="Embedding dimensionality", tickvals=list(dims))
    fig.update_yaxes(title_text="Stress-1 (SMACOF)")
    fig.update_layout(
        template="plotly_white",
        title="Weighted-SMACOF stress vs. embedding dimensionality",
        height=420, margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return fig




def fig_cross_cohort_agreement(agreement_by_x: dict[str, dict], dims: list[int]) -> go.Figure:
    """3 traces (Kendall tau, Spearman rho, 1 - Procrustes m^2) of pre- vs.
    post-cohort structural agreement on a shared canvas, x-axis = 'raw RDM'
    plus each embedding dimensionality. Procrustes has no value at 'raw RDM'
    (undefined on an incomplete RDM -- only Kendall/Spearman, restricted to
    jointly-covered pairs, are defined there)."""
    x_labels = ["raw RDM"] + [str(d) for d in dims]
    kendall = [agreement_by_x[k]["kendall"] for k in x_labels]
    spearman = [agreement_by_x[k]["spearman"] for k in x_labels]
    one_minus_m2 = [
        (1 - agreement_by_x[k]["procrustes_m2"]) if agreement_by_x[k].get("procrustes_m2") is not None else None
        for k in x_labels
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=kendall, mode="lines+markers", **stat_scatter_kwargs("kendall")))
    fig.add_trace(go.Scatter(x=x_labels, y=spearman, mode="lines+markers", **stat_scatter_kwargs("spearman")))
    fig.add_trace(go.Scatter(x=x_labels, y=one_minus_m2, mode="lines+markers", **stat_scatter_kwargs("one_minus_m2")))

    fig.update_xaxes(title_text="Embedding dimensionality", type="category")
    fig.update_yaxes(title_text="Agreement (bigger = better)")
    fig.update_layout(
        template="plotly_white",
        title="Pre- vs. post-SHINE structural agreement",
        height=420, margin={"l": 70, "r": 40, "t": 70, "b": 60},
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Cohort-label permutation test of pre- vs post-cohort RDM agreement
# ---------------------------------------------------------------------------


def parse_subject_trials(
    df_trials: pd.DataFrame, participant_ids: dict[str, Sequence[str]],
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], int]:
    """Parse every main trial once per subject (each cohort's subjects with
    their own SHINE variant's image paths), returning ({subject_id: sparse
    trials}, n_conds) -- the input rdm_from_subject_trials() aggregates."""
    image_to_idx, n_conds = image_manifest_index()
    trials_by_subject: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for cohort, ids in participant_ids.items():
        trials_by_subject.update(_parse_trials_by_participant(df_trials, ids, f"{cohort}_shine", image_to_idx, n_conds))
    return trials_by_subject, n_conds


def rdm_from_subject_trials(
    subject_ids: Sequence[str], trials_by_subject: dict[str, list[tuple[np.ndarray, np.ndarray]]], n_conds: int,
) -> np.ndarray:
    """One condensed behavioral RDM ("simple" aggregation, the project default)
    from the pooled trials of *subject_ids*. Bind the other two arguments with
    functools.partial to get the rdm_builder permutation_test_cohort_agreement() expects."""
    pooled = [trial for pid in subject_ids for trial in trials_by_subject[pid]]
    rdm, _ = _combine(pooled, n_conds, "simple")
    return rdm


def _init_worker(rdm_builder: RdmBuilder) -> None:
    global _worker_builder
    _worker_builder = rdm_builder


def _split_agreement(split: tuple[list[str], list[str]]) -> tuple[float, float]:
    ids_a, ids_b = split
    agreement = masked_rank_correlation(_worker_builder(ids_a), _worker_builder(ids_b))
    return agreement["kendall"], agreement["spearman"]


def _fingerprint(ids_a: Sequence[str], ids_b: Sequence[str], design: int | str, seed: int) -> str:
    payload = "|".join(ids_a) + "#" + "|".join(ids_b) + f"#{design}#{seed}"
    return hashlib.sha1(payload.encode()).hexdigest()


def _run_tasks(func: Callable, tasks: list, rdm_builder: RdmBuilder, n_jobs: int) -> np.ndarray:
    if n_jobs == 1:
        _init_worker(rdm_builder)
        return np.array([func(task) for task in tasks])
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(rdm_builder,)) as executor:
        return np.array(list(executor.map(func, tasks, chunksize=4)))


def _random_splits(
    pooled_ids: Sequence[str], n_a: int, n_splits: int, rng: np.random.Generator,
) -> list[tuple[list[str], list[str]]]:
    splits = []
    for _ in range(n_splits):
        shuffled = rng.permutation(np.array(pooled_ids)).tolist()
        splits.append((shuffled[:n_a], shuffled[n_a:]))
    return splits


def _draw_null(
    rdm_builder: RdmBuilder, ids_a: list[str], ids_b: list[str], n_perm: int, seed: int, n_jobs: int,
) -> np.ndarray:
    splits = _random_splits(ids_a + ids_b, len(ids_a), n_perm, np.random.default_rng(seed))
    return _run_tasks(_split_agreement, splits, rdm_builder, n_jobs)


def permutation_test_cohort_agreement(
    rdm_builder: RdmBuilder, ids_a: Sequence[str], ids_b: Sequence[str], *,
    n_perm: int = 500, seed: int = 0, n_jobs: int = 1, cache_path: Path | None = None,
) -> dict:
    """Are cohorts A and B less alike than exchangeable cohorts would be? The
    observed Kendall tau / Spearman rho between the two cohorts' RDMs (on
    jointly covered pairs) is compared with the same statistic for `n_perm`
    random re-assignments of the pooled subjects' cohort labels (group sizes
    kept). `rdm_builder(subject_ids)` returns one condensed RDM from a list of
    subject ids; one-sided p = (1 + #{null <= observed}) / (1 + n_perm).

    Tests for a difference, not for equivalence. With `cache_path`, the null
    draws are stored in an .npz keyed by a fingerprint of the subject ids,
    n_perm and seed, and reloaded on a matching re-run.

    Returns {"kendall"/"spearman": {observed, null_mean, null_low, null_high, p},
    "null": (n_perm, 2) array of [kendall, spearman] draws, "n_perm", "seed"}."""
    observed = masked_rank_correlation(rdm_builder(list(ids_a)), rdm_builder(list(ids_b)))
    null = cached_array(
        cache_path, _fingerprint(ids_a, ids_b, n_perm, seed),
        lambda: _draw_null(rdm_builder, list(ids_a), list(ids_b), n_perm, seed, n_jobs),
    )

    result = {"n_perm": n_perm, "seed": seed, "null": null}
    for column, key in enumerate(("kendall", "spearman")):
        draws = null[:, column]
        low, high = np.quantile(draws, [0.025, 0.975])
        result[key] = {
            "observed": observed[key], "null_mean": float(draws.mean()), "null_low": float(low),
            "null_high": float(high), "p": float((1 + np.sum(draws <= observed[key])) / (1 + n_perm)),
        }
    return result


def fig_permutation_null(result: dict, statistic: str, *, statistic_label: str) -> go.Figure:
    """Histogram of the label-permutation null for one statistic ("kendall" or
    "spearman") with the observed pre-vs-post value marked."""
    column = ("kendall", "spearman").index(statistic)
    summary = result[statistic]
    fig = go.Figure(go.Histogram(
        x=result["null"][:, column], nbinsx=40, marker_color=NULL_COLOR, name="Shuffled cohort labels",
    ))
    fig.add_vline(
        x=summary["observed"], line={"color": OBSERVED_COLOR, "width": 3},
        annotation_text=f"Observed {summary['observed']:.3f} (p={summary['p']:.3f})", annotation_position="top",
    )
    fig.update_xaxes(title_text=f"{statistic_label} between the two groups' RDMs")
    fig.update_yaxes(title_text="Permutations")
    fig.update_layout(
        template="plotly_white", title=f"Pre vs. post {statistic_label} against the label-permutation null",
        height=380, margin={"l": 70, "r": 40, "t": 70, "b": 60}, showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Within- vs between-cohort agreement at a matched group size
# ---------------------------------------------------------------------------

_WITHIN_BETWEEN_CONDITIONS = ("within_a", "within_b", "between")


def _within_between_draw(task: tuple[list[str], list[str], list[str], list[str]]) -> np.ndarray:
    a_first, a_second, b_first, b_second = task
    rdms = {key: _worker_builder(ids) for key, ids in (
        ("a1", a_first), ("a2", a_second), ("b1", b_first), ("b2", b_second),
    )}

    def agreement(left: str, right: str) -> np.ndarray:
        result = masked_rank_correlation(rdms[left], rdms[right])
        return np.array([result["kendall"], result["spearman"]])

    between = (agreement("a1", "b1") + agreement("a2", "b2")) / 2
    return np.stack([agreement("a1", "a2"), agreement("b1", "b2"), between])


def _within_between_tasks(
    ids_a: Sequence[str], ids_b: Sequence[str], group_size: int, n_draws: int, rng: np.random.Generator,
) -> list[tuple[list[str], list[str], list[str], list[str]]]:
    tasks = []
    for _ in range(n_draws):
        shuffled_a, shuffled_b = rng.permutation(np.array(ids_a)).tolist(), rng.permutation(np.array(ids_b)).tolist()
        tasks.append((
            shuffled_a[:group_size], shuffled_a[group_size:2 * group_size],
            shuffled_b[:group_size], shuffled_b[group_size:2 * group_size],
        ))
    return tasks


def within_between_agreement(
    rdm_builder: RdmBuilder, ids_a: Sequence[str], ids_b: Sequence[str], *,
    group_size: int | None = None, n_draws: int = 100, seed: int = 0, n_jobs: int = 1,
    cache_path: Path | None = None,
) -> dict:
    """Rank agreement between two disjoint groups of `group_size` subjects, within cohort A, within cohort B and
    between the cohorts, so that all three come from groups of the same size (default: half of the smaller
    cohort). Each draw splits each cohort at random into two groups; "within_a"/"within_b" correlate a cohort's two
    groups, and "between" averages the correlations of cohort A's group 1 with cohort B's group 1 and group 2 with
    group 2. All correlations are Kendall tau / Spearman rho on jointly covered pairs.

    The within-cohort values are the noise ceiling for the between-cohort value at that group size (the logic of
    the split-half bound, without the Spearman-Brown extrapolation to the full sample, which does not hold for
    these sparse RDMs). The draws reuse the same subjects, so their spread is descriptive, not sampling
    uncertainty. With `cache_path` the draws are cached under a fingerprint of the subject ids and design.

    Returns {condition: {"kendall"/"spearman": {mean, low, high}}} (low/high: 2.5th/97.5th percentile over draws),
    plus "draws" (n_draws, 3, 2) in the order of "conditions", "group_size", "n_draws" and "seed"."""
    group_size = group_size or min(len(ids_a), len(ids_b)) // 2
    if 2 * group_size > min(len(ids_a), len(ids_b)):
        raise ValueError(f"group_size {group_size} needs {2 * group_size} subjects per cohort")
    draws = cached_array(
        cache_path, _fingerprint(ids_a, ids_b, f"within_between_g{group_size}_n{n_draws}", seed),
        lambda: _run_tasks(
            _within_between_draw,
            _within_between_tasks(ids_a, ids_b, group_size, n_draws, np.random.default_rng(seed)),
            rdm_builder, n_jobs,
        ),
    )
    result = {
        "conditions": _WITHIN_BETWEEN_CONDITIONS, "draws": draws,
        "group_size": group_size, "n_draws": n_draws, "seed": seed,
    }
    for row, condition in enumerate(_WITHIN_BETWEEN_CONDITIONS):
        result[condition] = {}
        for column, statistic in enumerate(("kendall", "spearman")):
            low, high = np.quantile(draws[:, row, column], [0.025, 0.975])
            result[condition][statistic] = {
                "mean": float(draws[:, row, column].mean()), "low": float(low), "high": float(high),
            }
    return result


def fig_within_between(result: dict, statistic: str, *, statistic_label: str) -> go.Figure:
    """Violins of the within-pre, within-post and between-cohort agreement of two groups of the same size
    (one value per random split), with the mean marked."""
    column = ("kendall", "spearman").index(statistic)
    first, second = COHORTS
    labels = {
        "within_a": f"{cohort_label(first)} vs. {cohort_label(first)}",
        "within_b": f"{cohort_label(second)} vs. {cohort_label(second)}",
        "between": f"{cohort_label(first)} vs. {cohort_label(second)}",
    }
    line_colors = {
        "within_a": cohort_colors(first)["main"], "within_b": cohort_colors(second)["main"], "between": OBSERVED_COLOR,
    }
    fill_colors = {
        "within_a": cohort_colors(first)["violin"], "within_b": cohort_colors(second)["violin"], "between": NULL_COLOR,
    }
    fig = go.Figure()
    for row, condition in enumerate(result["conditions"]):
        fig.add_trace(go.Violin(
            y=result["draws"][:, row, column], name=labels[condition], line_color=line_colors[condition],
            fillcolor=fill_colors[condition], box_visible=True, meanline_visible=True, points=False, showlegend=False,
        ))
    fig.update_yaxes(title_text=f"{statistic_label} between two groups of {result['group_size']} subjects")
    fig.update_layout(
        template="plotly_white",
        title=f"Within- vs. between-cohort agreement at a matched group size ({result['n_draws']} random splits)",
        height=420, margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return fig


# ---------------------------------------------------------------------------
# Equivalence of pre- vs post-cohort RDM agreement
# ---------------------------------------------------------------------------


def _deviation_from_exchangeable(task: tuple) -> tuple[float, float]:
    ids_a, ids_b, splits = task
    observed = masked_rank_correlation(_worker_builder(ids_a), _worker_builder(ids_b))
    null = np.array([_split_agreement(split) for split in splits])
    return observed["kendall"] - null[:, 0].mean(), observed["spearman"] - null[:, 1].mean()


def equivalence_test_cohort_agreement(
    rdm_builder: RdmBuilder, ids_a: Sequence[str], ids_b: Sequence[str], permutation_result: dict, *,
    n_groups: int = 20, n_null: int = 10, alpha: float = 0.05, power: float = 0.8, step: float = 0.01,
    seed: int = 0, n_jobs: int = 1, cache_path: Path | None = None,
) -> dict[str, dict]:
    """Are cohorts A and B's RDMs as alike as exchangeable cohorts would be, to
    within a bound? The effect is D = observed agreement minus the mean of the
    label-permutation null (`permutation_result`, from
    permutation_test_cohort_agreement()), per statistic ("kendall", "spearman").

    The CI of D comes from a delete-a-group jackknife over subjects: each
    cohort's subjects are split into `n_groups` random groups, one group per
    cohort is left out in turn, and D is recomputed on the remaining subjects
    against a fresh `n_null`-draw permutation null of those subjects; the
    (1 - 2*alpha) CI is D +/- z * SE_jackknife. The permutation null's own
    spread is not used, because every shuffle reuses the same subjects and so
    understates sampling variability; a with-replacement bootstrap is avoided
    because duplicated subjects would end up in both shuffled groups.

    The bound follows the project rule: the smallest bound at which a true
    D = 0 is declared equivalent with probability `power`,
    tost_bound_multiplier(alpha, power) * SE (z_(1-alpha) + z_((1+power)/2)), rounded up to a multiple of `step` (in
    tau/rho units).

    Returns {"kendall"/"spearman": result dict} in the format of
    analysis.utils.equivalence's tests (effect, ci_low, ci_high, bound,
    equivalent), plus "se" and "sensitivity"."""
    ids_a, ids_b = list(ids_a), list(ids_b)

    def jackknife_deviations() -> np.ndarray:
        rng = np.random.default_rng(seed)
        groups_a = np.array_split(rng.permutation(ids_a), n_groups)
        groups_b = np.array_split(rng.permutation(ids_b), n_groups)
        tasks = []
        for g in range(n_groups):
            kept_a = [i for h, group in enumerate(groups_a) if h != g for i in group.tolist()]
            kept_b = [i for h, group in enumerate(groups_b) if h != g for i in group.tolist()]
            tasks.append((kept_a, kept_b, _random_splits(kept_a + kept_b, len(kept_a), n_null, rng)))
        return _run_tasks(_deviation_from_exchangeable, tasks, rdm_builder, n_jobs)

    replicates = cached_array(
        cache_path, _fingerprint(ids_a, ids_b, f"jackknife{n_groups}x{n_null}", seed), jackknife_deviations,
    )

    z_ci = float(norm.ppf(1 - alpha))
    results = {}
    for column, key in enumerate(("kendall", "spearman")):
        effect = permutation_result[key]["observed"] - permutation_result[key]["null_mean"]
        deviations = replicates[:, column]
        se = float(np.sqrt((n_groups - 1) / n_groups * np.sum((deviations - deviations.mean()) ** 2)))
        low, high = effect - z_ci * se, effect + z_ci * se
        sensitivity = tost_bound_multiplier(alpha, power) * se
        bound = resolvable_bound(sensitivity, step=step)
        results[key] = {
            "comparison": f"observed - exchangeable expectation ({key})", "n (A)": len(ids_a), "n (B)": len(ids_b),
            "effect": round(float(effect), 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4),
            "bound": bound, "equivalent": bool(-bound < low and high < bound),
            "se": round(se, 5), "sensitivity": round(sensitivity, 5),
        }
    return results
