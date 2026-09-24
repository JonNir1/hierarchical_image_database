"""
Pre- vs post-SHINE cohort comparison, structure layer (analysis/behavioral/
cohort_structure.ipynb): MDS-based structural comparison of the two cohorts'
behavioral RDMs (eigenvalue scree, weighted-SMACOF stress, cross-cohort
agreement), the cohort-label permutation test of pre-vs-post RDM agreement, and
its equivalence counterpart (delete-a-group jackknife CI of observed minus the
exchangeable-null expectation).

MDS-embedding logic itself (classical eigenspectrum, cached SMACOF sweep,
cross-embedding comparison) lives in analysis/rdms/mds.py.

Usage (from repo root):
    from analysis.behavioral.cohort_structure import (
        MDS_CACHE_DIR, PERMUTATION_CACHE_DIR, fig_eigenvalue_scree, fig_stress_scree,
        fig_cross_cohort_agreement, permutation_test_cohort_agreement, fig_permutation_null,
        equivalence_test_cohort_agreement,
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
from scipy.stats import norm

from analysis.behavioral.cohort_comparison import COHORTS, cohort_colors, cohort_label
from analysis.behavioral.noise_ceiling import _parse_trials_by_participant
from analysis.rdms.behavioral import _combine
from analysis.rdms.common import REPO_ROOT, image_manifest_index
from analysis.utils.equivalence import resolvable_bound
from analysis.utils.stats import masked_rank_correlation

MDS_CACHE_DIR = REPO_ROOT / "analysis" / "results" / "behavioral" / "mds_cache"
PERMUTATION_CACHE_DIR = REPO_ROOT / "analysis" / "results" / "behavioral" / "permutation_cache"

RdmBuilder = Callable[[Sequence[str]], np.ndarray]

_worker_builder: RdmBuilder | None = None


def fig_eigenvalue_scree(eig_by_cohort: dict[str, np.ndarray], max_dims: int) -> go.Figure:
    """Line+marker scree plot of the first *max_dims* classical-MDS
    eigenvalues, one trace per cohort. See
    analysis.rdms.mds.classical_mds_eigenspectrum()'s caveat -- qualitative
    view only, not a validated dimensionality estimate at this RDM's
    coverage level."""
    fig = go.Figure()
    x = list(range(1, max_dims + 1))
    for cohort in COHORTS:
        cc = cohort_colors(cohort)
        fig.add_trace(go.Scatter(
            x=x, y=eig_by_cohort[cohort][:max_dims].tolist(),
            mode="lines+markers", line={"color": cc["main"], "width": 2.5},
            marker={"size": 8, "color": cc["main"]}, name=cohort_label(cohort),
        ))
    fig.update_xaxes(title_text="Component", tickvals=x)
    fig.update_yaxes(title_text="Eigenvalue")
    fig.update_layout(
        template="plotly_white",
        title=f"Classical-MDS eigenvalue scree (mean-imputed -- qualitative only, first {max_dims} components)",
        height=420, margin={"l": 70, "r": 40, "t": 70, "b": 60},
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
            line={"color": cc["main"], "width": 2.5}, marker={"size": 8, "color": cc["main"]},
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


_AGREEMENT_COLORS = {"kendall": "#2a78d6", "spearman": "#1baf7a", "one_minus_m2": "#4a3aa7"}


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
    fig.add_trace(go.Scatter(x=x_labels, y=kendall, mode="lines+markers", name="Kendall tau", line={"color": _AGREEMENT_COLORS["kendall"], "width": 2.5}, marker={"size": 8}))
    fig.add_trace(go.Scatter(x=x_labels, y=spearman, mode="lines+markers", name="Spearman rho", line={"color": _AGREEMENT_COLORS["spearman"], "width": 2.5}, marker={"size": 8}))
    fig.add_trace(go.Scatter(x=x_labels, y=one_minus_m2, mode="lines+markers", name="1 - Procrustes m2", line={"color": _AGREEMENT_COLORS["one_minus_m2"], "width": 2.5}, marker={"size": 8}))

    fig.update_xaxes(title_text="Comparison", type="category")
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


def _load_cached_null(cache_path: Path | None, fingerprint: str) -> np.ndarray | None:
    if cache_path is None or not Path(cache_path).exists():
        return None
    with np.load(cache_path) as cached:
        if str(cached["fingerprint"]) != fingerprint:
            return None
        return cached["null"]


def _save_cached_null(cache_path: Path | None, fingerprint: str, null: np.ndarray) -> None:
    if cache_path is None:
        return
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, null=null, fingerprint=fingerprint)


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
    fingerprint = _fingerprint(ids_a, ids_b, n_perm, seed)
    null = _load_cached_null(cache_path, fingerprint)
    observed = masked_rank_correlation(rdm_builder(list(ids_a)), rdm_builder(list(ids_b)))
    if null is None:
        null = _draw_null(rdm_builder, list(ids_a), list(ids_b), n_perm, seed, n_jobs)
        _save_cached_null(cache_path, fingerprint, null)

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
        x=result["null"][:, column], nbinsx=40, marker_color="rgba(120,120,120,0.6)", name="Shuffled cohort labels",
    ))
    fig.add_vline(
        x=summary["observed"], line={"color": "#eb6834", "width": 3},
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
    (z_(1-alpha) + z_power) * SE, rounded up to a multiple of `step` (in
    tau/rho units).

    Returns {"kendall"/"spearman": result dict} in the format of
    analysis.utils.equivalence's tests (effect, ci_low, ci_high, bound,
    equivalent), plus "se" and "sensitivity"."""
    ids_a, ids_b = list(ids_a), list(ids_b)
    fingerprint = _fingerprint(ids_a, ids_b, f"jackknife{n_groups}x{n_null}", seed)
    replicates = _load_cached_null(cache_path, fingerprint)
    if replicates is None:
        rng = np.random.default_rng(seed)
        groups_a = np.array_split(rng.permutation(ids_a), n_groups)
        groups_b = np.array_split(rng.permutation(ids_b), n_groups)
        tasks = []
        for g in range(n_groups):
            kept_a = [i for h, group in enumerate(groups_a) if h != g for i in group.tolist()]
            kept_b = [i for h, group in enumerate(groups_b) if h != g for i in group.tolist()]
            tasks.append((kept_a, kept_b, _random_splits(kept_a + kept_b, len(kept_a), n_null, rng)))
        replicates = _run_tasks(_deviation_from_exchangeable, tasks, rdm_builder, n_jobs)
        _save_cached_null(cache_path, fingerprint, replicates)

    z_ci, z_power = float(norm.ppf(1 - alpha)), float(norm.ppf(power))
    results = {}
    for column, key in enumerate(("kendall", "spearman")):
        effect = permutation_result[key]["observed"] - permutation_result[key]["null_mean"]
        deviations = replicates[:, column]
        se = float(np.sqrt((n_groups - 1) / n_groups * np.sum((deviations - deviations.mean()) ** 2)))
        low, high = effect - z_ci * se, effect + z_ci * se
        sensitivity = (z_ci + z_power) * se
        bound = resolvable_bound(sensitivity, step=step)
        results[key] = {
            "comparison": f"observed - exchangeable expectation ({key})", "n (A)": len(ids_a), "n (B)": len(ids_b),
            "effect": round(float(effect), 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4),
            "bound": bound, "equivalent": bool(-bound < low and high < bound),
            "se": round(se, 5), "sensitivity": round(sensitivity, 5),
        }
    return results
