"""
Pre- vs post-SHINE cohort comparison, structure layer (analysis/behavioral/
cohort_structure.ipynb): MDS-based structural comparison of the two cohorts'
behavioral RDMs (eigenvalue scree, weighted-SMACOF stress, cross-cohort
agreement).

MDS-embedding logic itself (classical eigenspectrum, cached SMACOF sweep,
cross-embedding comparison) lives in analysis/rdms/mds.py.

Usage (from repo root):
    from analysis.behavioral.cohort_structure import (
        MDS_CACHE_DIR, fig_eigenvalue_scree, fig_stress_scree, fig_cross_cohort_agreement,
    )
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from analysis.behavioral.cohort_comparison import COHORTS, cohort_colors, cohort_label
from analysis.rdms.common import REPO_ROOT

MDS_CACHE_DIR = REPO_ROOT / "analysis" / "results" / "behavioral" / "mds_cache"


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
