"""
Pre- vs post-SHINE cohort comparison, structure layer (analysis/behavioral/
cohort_structure.ipynb): MDS-based structural comparison of the two cohorts'
behavioral RDMs (eigenvalue scree, weighted-SMACOF stress, cross-cohort
agreement) and the cohort-label permutation test of pre-vs-post RDM agreement.

MDS-embedding logic itself (classical eigenspectrum, cached SMACOF sweep,
cross-embedding comparison) lives in analysis/rdms/mds.py.

Usage (from repo root):
    from analysis.behavioral.cohort_structure import (
        MDS_CACHE_DIR, PERMUTATION_CACHE_DIR, fig_eigenvalue_scree, fig_stress_scree,
        fig_cross_cohort_agreement, permutation_test_cohort_agreement, fig_permutation_null,
    )
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import plotly.graph_objects as go

import pandas as pd

from analysis.behavioral.cohort_comparison import COHORTS, cohort_colors, cohort_label
from analysis.behavioral.noise_ceiling import _parse_trials_by_participant
from analysis.rdms.behavioral import _combine
from analysis.rdms.common import REPO_ROOT, image_manifest_index
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


def _fingerprint(ids_a: Sequence[str], ids_b: Sequence[str], n_perm: int, seed: int) -> str:
    payload = "|".join(ids_a) + "#" + "|".join(ids_b) + f"#{n_perm}#{seed}"
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


def _draw_null(
    rdm_builder: RdmBuilder, ids_a: list[str], ids_b: list[str], n_perm: int, seed: int, n_jobs: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pooled = np.array(ids_a + ids_b)
    splits = []
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled).tolist()
        splits.append((shuffled[:len(ids_a)], shuffled[len(ids_a):]))
    if n_jobs == 1:
        _init_worker(rdm_builder)
        return np.array([_split_agreement(split) for split in splits])
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(rdm_builder,)) as executor:
        return np.array(list(executor.map(_split_agreement, splits, chunksize=4)))


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
