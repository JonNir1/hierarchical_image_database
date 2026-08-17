"""Every figure in the v5 report, one function per named placeholder.

The prose in ``report_v5.src.html`` is hand-written; the figures are not. Each function here reads
the run's CSVs and returns a plotly Figure, so a figure cannot drift from the tables behind it even
though the surrounding text can.

Register a figure by adding it to :data:`FIGURES`. ``assemble.py`` fails if the source references a
name that is not registered, or if a registered name is never used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ARMS = {0.0: "random", 1.0: "constrained-MacDonald"}
COLOUR = {"random": "#888888", "constrained-MacDonald": "#1f77b4",
          "designed_unconstrained": "#9467bd", "pilot": "#d62728", "sim": "#1f77b4"}
LINKAGE_COLOUR = {"average": "#1f77b4", "ward": "#2ca02c", "complete": "#ff7f0e"}
PLOT_BG = "rgba(0,0,0,0)"
HIGH_K = 150


# --------------------------------------------------------------------------- loading
class Run:
    """The run's tables, loaded once and shared by every figure."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self._cache: Dict[str, pd.DataFrame] = {}

    def table(self, name: str, subdir: str = "out") -> pd.DataFrame:
        if name not in self._cache:
            path = self.dir / subdir / f"{name}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found; the report needs it for a figure")
            frame = pd.read_csv(path)
            if "allocation_mode" in frame.columns and "arm" not in frame.columns:
                frame["arm"] = frame["allocation_mode"].map(ARMS)
            self._cache[name] = frame
        return self._cache[name]

    def gt(self, name: str) -> pd.DataFrame:
        return self.table(name, subdir="gt_diagnostics")

    def stage1(self, name: str) -> pd.DataFrame:
        return self.table(name, subdir="gt")

    def meta(self) -> pd.DataFrame:
        if "_meta" not in self._cache:
            self._cache["_meta"] = pd.read_csv(self.dir / "mds_store" / "meta.csv",
                                               float_precision="round_trip")
        return self._cache["_meta"]


# --------------------------------------------------------------------------- shared styling
def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)), height=height,
        margin=dict(l=60, r=30, t=50, b=50), paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(size=12), legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="closest")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def _by_arm(frame: pd.DataFrame, value: str, title: str, ylab: str,
            hline: Optional[float] = None, height: int = 380) -> go.Figure:
    """Mean +/- SEM against N, one trace per allocation arm. The workhorse of the report."""
    fig = go.Figure()
    for arm in ("random", "constrained-MacDonald"):
        sub = frame[frame["arm"] == arm]
        if sub.empty:
            continue
        agg = sub.groupby("num_subjects")[value].agg(["mean", "std", "count"]).reset_index()
        fig.add_trace(go.Scatter(
            x=agg["num_subjects"], y=agg["mean"], mode="lines+markers", name=arm,
            line=dict(color=COLOUR[arm], width=2), marker=dict(size=8),
            error_y=dict(type="data", visible=True, thickness=1,
                         array=agg["std"] / np.sqrt(agg["count"].clip(lower=1)))))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(dash="dot", color="rgba(128,128,128,0.7)"))
    fig.update_xaxes(title_text="participants (N)", type="log",
                     tickvals=sorted(frame["num_subjects"].unique()))
    fig.update_yaxes(title_text=ylab)
    return _style(fig, title, height)


# --------------------------------------------------------------------------- ground truth
def gt_dimensionality_scan(run: Run) -> go.Figure:
    scan, cv = run.stage1("scan_summary"), run.stage1("cv_summary")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scan["ndim"], y=scan["spearman_mean"], mode="lines+markers", name="split-half",
        line=dict(color="#1f77b4", width=2), marker=dict(size=8),
        error_y=dict(type="data", array=scan["spearman_sem"], visible=True, thickness=1)))
    fig.add_trace(go.Scatter(
        x=cv["ndim"], y=cv["spearman_mean"], mode="lines+markers", name="leave-k-out",
        line=dict(color="#2ca02c", width=2, dash="dot"), marker=dict(size=7),
        error_y=dict(type="data", array=cv["spearman_sem"], visible=True, thickness=1)))
    fig.update_xaxes(title_text="ground-truth dimensionality")
    fig.update_yaxes(title_text="Spearman ρ between independent halves")
    return _style(fig, "Split-half agreement peaks at D=3 and is flat across 3-5")


def gt_topk_by_ndim(run: Run) -> go.Figure:
    scan = run.stage1("scan_summary")
    fig = go.Figure(go.Scatter(x=scan["ndim"], y=scan["topk_jaccard_mean"], mode="lines+markers",
                               line=dict(color="#ff7f0e", width=2), marker=dict(size=8),
                               error_y=dict(type="data", array=scan["topk_jaccard_sem"],
                                            visible=True, thickness=1)))
    fig.update_xaxes(title_text="ground-truth dimensionality")
    fig.update_yaxes(title_text="top-k Jaccard between halves")
    return _style(fig, "Closest-pair agreement keeps rising to D=20, unlike overall agreement")


def gt_vs_noise_ceiling(run: Run) -> go.Figure:
    # gt_vs_raw already carries ceiling_full: `gt_diagnostics.diagnose` joins it in so that
    # frac_of_ceiling can be derived. Re-joining would only produce suffixed duplicates.
    merged = run.gt("gt_vs_raw")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=merged["level_name"], y=merged["spearman"], marker_color="#1f77b4",
                         name="ground truth vs pooled ratings (in-sample)"))
    fig.add_trace(go.Bar(x=merged["level_name"], y=merged["ceiling_full"], marker_color="#d62728",
                         name="noise ceiling (ratings vs themselves)"))
    fig.update_xaxes(title_text="semantic relation, coarse to fine")
    fig.update_yaxes(title_text="Spearman ρ")
    return _style(fig, "The embedding exceeds the ceiling its own data sets")


def gt_semantic_gradient(run: Run) -> go.Figure:
    g = run.gt("gt_gradient")
    rel = g["mean_distance"] / g["mean_distance"].iloc[0]
    colours = ["#1f77b4"] * len(g)
    colours[-1] = "#d62728"
    fig = go.Figure(go.Bar(x=g["level_name"], y=rel, marker_color=colours,
                           text=[f"{v:.3f}" for v in rel], textposition="outside"))
    fig.update_xaxes(title_text="semantic relation, coarse to fine")
    fig.update_yaxes(title_text="mean distance / unrelated-pair distance")
    return _style(fig, "Monotone for five levels; depth-5 (red) inverts")


# --------------------------------------------------------------------------- realism
def realism_calibration(run: Run) -> go.Figure:
    """Target vs achieved for the two observables the noise model was inverted against.

    Read from calibration.json rather than retyped, so the figure cannot drift from the fit.
    """
    import json
    cal = json.loads((run.dir / "calibration" / "calibration.json").read_text())
    labels = ["between-subject agreement<br>(mean pairwise Spearman ρ)",
              "within-subject test-retest<br>(Spearman ρ on repeats)"]
    empirical = [cal["empirical_agreement"], cal["target_test_retest"]]
    achieved = [cal["dispersion_achieved"], cal["achieved_tr_unscreened"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=empirical, name="empirical (pilot)", marker_color="#d62728",
                         text=[f"{v:.3f}" for v in empirical], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=achieved, name="achieved (simulated)", marker_color="#1f77b4",
                         text=[f"{v:.3f}" for v in achieved], textposition="outside"))
    fig.update_yaxes(title_text="Spearman ρ", range=[0, max(empirical + achieved) * 1.25])
    return _style(fig, "One target is hit; the other is undershot")


def realism_noise_vs_distance(run: Run) -> go.Figure:
    curves = run.table("noise_vs_distance")
    fig = go.Figure()
    for src, label in (("pilot", "pilot participants"), ("sim", "simulated participants")):
        sub = curves[curves["source"] == src]
        fig.add_trace(go.Scatter(x=sub["mean_pair_distance"], y=sub["rmse"], mode="lines+markers",
                                 name=label, line=dict(color=COLOUR[src], width=2),
                                 error_y=dict(type="data", array=sub["sem_rmse"], visible=True,
                                              thickness=1)))
    fig.update_xaxes(title_text="mean placed distance of the pair")
    fig.update_yaxes(title_text="RMSE between the two placements")
    return _style(fig, "Both curves rise then fall: the inverted U the model was not fitted to")


def realism_semantic_gradient(run: Run) -> go.Figure:
    g = run.table("validity_gradient")
    g = g[g["arm"] == "random"] if "arm" in g.columns else g
    fig = go.Figure()
    for src, label in (("pilot", "pilot participants"), ("sim", "simulated participants")):
        col = f"mean_distance_{src}"
        if col in g.columns:
            fig.add_trace(go.Scatter(x=g["level_name"], y=g[col] / g[col].iloc[0],
                                     mode="lines+markers", name=label,
                                     line=dict(color=COLOUR[src], width=2), marker=dict(size=8)))
    fig.update_xaxes(title_text="semantic relation, coarse to fine")
    fig.update_yaxes(title_text="mean distance / unrelated-pair distance")
    return _style(fig, "The model orders the levels correctly but under-separates the fine end")


# --------------------------------------------------------------------------- coverage
def coverage_by_n(run: Run) -> go.Figure:
    fig = _by_arm(run.table("coverage"), "pair_coverage",
                  "Share of the 262,450 image pairs judged by anyone", "pairs covered (%)")
    fig.add_hline(y=100, line=dict(dash="dot", color="rgba(128,128,128,0.6)"))
    return fig


def coverage_gain(run: Run) -> go.Figure:
    cov = run.table("coverage").groupby(["num_subjects", "arm"])["pair_coverage"].mean().unstack()
    gain = 100 * (cov["constrained-MacDonald"] - cov["random"]) / cov["random"]
    fig = go.Figure(go.Bar(x=[str(int(n)) for n in gain.index], y=gain.values,
                           marker_color="#1f77b4",
                           text=[f"{v:+.1f}%" for v in gain.values], textposition="outside"))
    fig.update_xaxes(title_text="participants (N)")
    fig.update_yaxes(title_text="extra pairs covered vs random (%)")
    return _style(fig, "The advantage peaks at N=75 and vanishes once random saturates")


def allocation_balance(run: Run) -> go.Figure:
    """Constrained vs unconstrained MacDonald vs random: the price of session-disjointness."""
    d = run.table("design_only")
    g = d.groupby(["num_subjects", "arm"])[["frac_pairs_covered", "reps_per_image_sd"]].mean()
    fig = go.Figure()
    for arm, label in (("random", "random"),
                       ("designed", "constrained MacDonald"),
                       ("designed_unconstrained", "unconstrained MacDonald")):
        if arm not in d["arm"].unique():
            continue
        sub = g.xs(arm, level="arm")
        fig.add_trace(go.Bar(x=[str(int(n)) for n in sub.index], y=sub["reps_per_image_sd"],
                             name=label,
                             marker_color=COLOUR.get("constrained-MacDonald" if arm == "designed"
                                                     else arm, "#9467bd")))
    fig.update_xaxes(title_text="participants (N)")
    fig.update_yaxes(title_text="SD of appearances per image")
    return _style(fig, "How evenly the 725 images are used (lower is more balanced)")


def constraint_cost(run: Run) -> go.Figure:
    d = run.table("design_only")
    g = d.groupby(["num_subjects", "arm"])["frac_pairs_covered"].mean().unstack()
    fig = go.Figure()
    for arm, label, dash in (("random", "random", "solid"),
                             ("designed", "constrained MacDonald", "solid"),
                             ("designed_unconstrained", "unconstrained MacDonald", "dot")):
        if arm not in g.columns:
            continue
        fig.add_trace(go.Scatter(x=g.index, y=100 * g[arm], mode="lines+markers", name=label,
                                 line=dict(width=2, dash=dash,
                                           color=COLOUR.get("constrained-MacDonald"
                                                            if arm == "designed" else arm,
                                                            "#9467bd"))))
    fig.update_xaxes(title_text="participants (N)", type="log", tickvals=sorted(g.index))
    fig.update_yaxes(title_text="pairs covered (%)")
    return _style(fig, "Session-disjointness costs almost nothing in coverage")


def mds_convergence(run: Run) -> go.Figure:
    """SMACOF outcome: how many fits converged, and the stress they converged to."""
    m = run.meta()
    ok = m[m["status"] == "success"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ok["stress"], nbinsx=60, marker_color="#1f77b4",
                               name=f"converged (n={len(ok):,})"))
    hit = m[m["status"] == "max_iters"]
    if len(hit):
        fig.add_trace(go.Histogram(x=hit["stress"], nbinsx=60, marker_color="#d62728",
                                   name=f"hit iteration cap (n={len(hit)})"))
    fig.update_xaxes(title_text="SMACOF stress at the returned solution")
    fig.update_yaxes(title_text="fits")
    fig.update_layout(barmode="overlay")
    return _style(fig, "Stress distribution across all fits")


# --------------------------------------------------------------------------- stability
def stability_raw(run: Run) -> go.Figure:
    return _by_arm(run.table("stability"), "spearman",
                   "Agreement between two cohorts' pooled ratings, before MDS",
                   "Spearman ρ")


def stability_embedding(run: Run) -> go.Figure:
    return _by_arm(run.table("embedding_stability"), "mean_spearman",
                   "Agreement between two cohorts' recovered embeddings",
                   "Spearman ρ between recovered distances")


def stability_procrustes(run: Run) -> go.Figure:
    return _by_arm(run.table("embedding_generalizability"), "mean_procrustes_m2",
                   "Procrustes m² between two cohorts' embeddings (lower is better)",
                   "Procrustes m²")


# --------------------------------------------------------------------------- closest pairs
def topk_between_cohorts(run: Run) -> go.Figure:
    return _by_arm(run.table("topk_jaccard"), "mean_jaccard",
                   "Do two cohorts agree on which pairs are closest?", "top-k Jaccard")


def recovery_recall(run: Run) -> go.Figure:
    return _by_arm(run.table("recovery_vs_gt"), "mean_recall",
                   "Recall of the ground truth's closest pairs", "recall@k")


def recovery_auc(run: Run) -> go.Figure:
    return _by_arm(run.table("recovery_vs_gt"), "mean_auc",
                   "Separating truly-close pairs from far ones", "AUC", hline=0.5)


def recovery_dprime(run: Run) -> go.Figure:
    return _by_arm(run.table("recovery_vs_gt"), "mean_dprime",
                   "Discriminability of the closest pairs", "d′")


# --------------------------------------------------------------------------- clusters
def cluster_vi_by_k(run: Run) -> go.Figure:
    curve = run.table("cluster_agreement").groupby(["linkage", "k"])[
        ["mean_vi_norm", "mean_ari"]].mean().reset_index()
    fig = go.Figure()
    for linkage, sub in curve.groupby("linkage"):
        fig.add_trace(go.Scatter(x=sub["k"], y=sub["mean_vi_norm"], mode="lines+markers",
                                 name=f"{linkage} (VI)",
                                 line=dict(color=LINKAGE_COLOUR[linkage], width=2)))
    fig.update_xaxes(title_text="clusters requested (k)", type="log",
                     tickvals=sorted(curve["k"].unique()))
    fig.update_yaxes(title_text="normalised VI (lower is more agreement)")
    fig.add_vrect(x0=HIGH_K, x1=curve["k"].max(), fillcolor="rgba(214,39,40,0.08)", line_width=0)
    return _style(fig, "Partition agreement degrades steadily as the cut gets finer")


def cluster_ari_by_k(run: Run) -> go.Figure:
    curve = run.table("cluster_agreement").groupby(["linkage", "k"])["mean_ari"].mean().reset_index()
    fig = go.Figure()
    for linkage, sub in curve.groupby("linkage"):
        fig.add_trace(go.Scatter(x=sub["k"], y=sub["mean_ari"], mode="lines+markers", name=linkage,
                                 line=dict(color=LINKAGE_COLOUR[linkage], width=2)))
    fig.update_xaxes(title_text="clusters requested (k)", type="log",
                     tickvals=sorted(curve["k"].unique()))
    fig.update_yaxes(title_text="adjusted Rand index")
    return _style(fig, "ARI tells the same story as VI on a chance-corrected scale")


def cluster_silhouette_by_k(run: Run) -> go.Figure:
    curve = run.table("cluster_agreement").groupby(["linkage", "k"])[
        "mean_sil_cross"].mean().reset_index()
    fig = go.Figure()
    for linkage, sub in curve.groupby("linkage"):
        fig.add_trace(go.Scatter(x=sub["k"], y=sub["mean_sil_cross"], mode="lines+markers",
                                 name=linkage, line=dict(color=LINKAGE_COLOUR[linkage], width=2)))
    fig.add_hline(y=0, line=dict(dash="dash", color="rgba(128,128,128,0.8)"))
    fig.add_vrect(x0=HIGH_K, x1=curve["k"].max(), fillcolor="rgba(214,39,40,0.08)", line_width=0,
                  annotation_text="least supported", annotation_position="top left")
    fig.update_xaxes(title_text="clusters requested (k)", type="log",
                     tickvals=sorted(curve["k"].unique()))
    fig.update_yaxes(title_text="cross-cohort silhouette")
    return _style(fig, "Above k≈12 the clusters no longer separate in another cohort's geometry")


def cluster_dendrogram_agreement(run: Run) -> go.Figure:
    d = run.table("dendrogram_agreement").groupby("linkage")[
        ["mean_baker_gamma", "mean_cophenetic_fidelity"]].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["linkage"], y=d["mean_baker_gamma"], name="Baker's γ",
                         marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=d["linkage"], y=d["mean_cophenetic_fidelity"],
                         name="cophenetic correlation", marker_color="#2ca02c"))
    fig.update_xaxes(title_text="linkage")
    fig.update_yaxes(title_text="correlation")
    return _style(fig, "Whole-tree agreement, independent of any cut")


def cluster_density(run: Run) -> go.Figure:
    g = run.table("density_agreement").groupby("min_cluster_size")[
        ["mean_frac_noise", "mean_noise_kappa"]].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["min_cluster_size"], y=g["mean_frac_noise"], mode="lines+markers",
                             name="share left unclustered", line=dict(color="#d62728", width=2)))
    fig.add_trace(go.Scatter(x=g["min_cluster_size"], y=g["mean_noise_kappa"], mode="lines+markers",
                             name="Cohen's κ on which images", line=dict(color="#1f77b4", width=2)))
    fig.update_xaxes(title_text="HDBSCAN min_cluster_size")
    fig.update_yaxes(title_text="proportion")
    return _style(fig, "Most images belong to no cluster, and cohorts barely agree on which")


FIGURES: Dict[str, Callable[[Run], go.Figure]] = {
    "gt_dimensionality_scan": gt_dimensionality_scan,
    "gt_topk_by_ndim": gt_topk_by_ndim,
    "gt_vs_noise_ceiling": gt_vs_noise_ceiling,
    "gt_semantic_gradient": gt_semantic_gradient,
    "realism_calibration": realism_calibration,
    "realism_noise_vs_distance": realism_noise_vs_distance,
    "realism_semantic_gradient": realism_semantic_gradient,
    "coverage_by_n": coverage_by_n,
    "coverage_gain": coverage_gain,
    "allocation_balance": allocation_balance,
    "constraint_cost": constraint_cost,
    "mds_convergence": mds_convergence,
    "stability_raw": stability_raw,
    "stability_embedding": stability_embedding,
    "stability_procrustes": stability_procrustes,
    "topk_between_cohorts": topk_between_cohorts,
    "recovery_recall": recovery_recall,
    "recovery_auc": recovery_auc,
    "recovery_dprime": recovery_dprime,
    "cluster_vi_by_k": cluster_vi_by_k,
    "cluster_ari_by_k": cluster_ari_by_k,
    "cluster_silhouette_by_k": cluster_silhouette_by_k,
    "cluster_dendrogram_agreement": cluster_dendrogram_agreement,
    "cluster_density": cluster_density,
}
