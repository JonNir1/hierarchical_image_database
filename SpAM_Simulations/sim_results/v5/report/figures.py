"""Every figure in the v5 report, one entry per named placeholder.

The prose in ``report_v5.src.html`` is hand-written; the figures are not. Each function here reads
the run's CSVs and returns a plotly Figure, so a figure cannot drift from the tables behind it even
though the surrounding text can.

Every figure carries a caption, rendered beneath it by ``assemble.py``. Captions state what the
error bars are, because a reader cannot otherwise tell whether a small bar means "precise" or
"we divided by a large n".

**Error bars are SD, not SEM.** The question these figures answer is "how much do our simulations
vary", not "how precisely do we know their mean". With hundreds of cells per point the SEM is
invisible and would imply a precision that is not the claim being made.

Register a figure in :data:`FIGURES`. ``assemble.py`` fails if the source references a name that is
not registered, or if a registered name is never used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

ARMS = {0.0: "random", 1.0: "constrained-MacDonald"}
COLOUR = {"random": "#888888", "constrained-MacDonald": "#1f77b4",
          "designed_unconstrained": "#9467bd", "pilot": "#d62728", "sim": "#1f77b4"}
LINKAGE_COLOUR = {"average": "#1f77b4", "ward": "#2ca02c", "complete": "#ff7f0e"}
PLOT_BG = "rgba(0,0,0,0)"
HIGH_K = 150

# "top-k" is always specified as a fraction of all C(725,2) pairs, so k is a count of pairs, not of
# images. Spelling both out keeps every axis label honest about what k actually is.
N_IMAGES = 725
N_PAIRS = N_IMAGES * (N_IMAGES - 1) // 2      # 262,450
TOP_FRAC_GT = 0.05                            # gt_construction.DEFAULT_TOP_FRAC
K_GT = round(TOP_FRAC_GT * N_PAIRS)


def k_label(frac: float) -> str:
    return f"top {frac:.0%} of pairs (k = {round(frac * N_PAIRS):,})"


SCREEN_LABEL = {-1.0: "none", 0.0: "ρ > 0", 0.1: "ρ > 0.1", 0.2: "ρ > 0.2"}

SD_NOTE = "Error bars are ±1 SD across every simulated cohort and swept setting at that N."


class Fig(NamedTuple):
    fn: Callable[["Run"], go.Figure]
    caption: str


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

    def calibration(self) -> dict:
        return json.loads((self.dir / "calibration" / "calibration.json").read_text())

    def meta(self) -> pd.DataFrame:
        if "_meta" not in self._cache:
            self._cache["_meta"] = pd.read_csv(self.dir / "mds_store" / "meta.csv",
                                               float_precision="round_trip")
        return self._cache["_meta"]


# --------------------------------------------------------------------------- shared styling
def _style(fig: go.Figure, title: str, height: int = 380, legend_rows: int = 1) -> go.Figure:
    """Shared layout, with the legend above the plot and the title above the legend.

    ``legend_rows`` is how many rows the legend actually wraps onto. Long entries wrap, and a
    wrapped legend grows upward into the title, so anything past a single row is moved below the
    plot instead of being squeezed in above it.
    """
    if legend_rows > 1:
        bottom = 60 + 22 * legend_rows
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)), height=height + bottom - 50,
            margin=dict(l=60, r=30, t=50, b=bottom), paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(size=12), hovermode="closest",
            legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    else:
        # The title is pinned to the container and the legend to the top of the plot area, so the
        # top margin holds both in that order and a wide legend cannot run under a centred title.
        fig.update_layout(
            title=dict(text=title, font=dict(size=14), yref="container", y=1.0, yanchor="top",
                       pad=dict(t=10)), height=height + 22,
            margin=dict(l=60, r=30, t=72, b=50), paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(size=12), legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
            hovermode="closest")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def _by_arm(frame: pd.DataFrame, value: str, title: str, ylab: str,
            hline: Optional[float] = None, height: int = 380) -> go.Figure:
    """Mean ±1 SD against N, one trace per allocation arm. The workhorse of the report."""
    fig = go.Figure()
    for arm in ("random", "constrained-MacDonald"):
        sub = frame[frame["arm"] == arm]
        if sub.empty:
            continue
        agg = sub.groupby("num_subjects")[value].agg(["mean", "std"]).reset_index()
        fig.add_trace(go.Scatter(
            x=agg["num_subjects"], y=agg["mean"], mode="lines+markers", name=arm,
            line=dict(color=COLOUR[arm], width=2), marker=dict(size=8),
            error_y=dict(type="data", array=agg["std"], visible=True, thickness=1)))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(dash="dot", color="rgba(128,128,128,0.7)"))
    fig.update_xaxes(title_text="participants (N)", type="log",
                     tickvals=sorted(frame["num_subjects"].unique()))
    fig.update_yaxes(title_text=ylab)
    return _style(fig, title, height)


def _by_arm_and_k(frame: pd.DataFrame, value: str, title: str, ylab: str,
                  hline: Optional[float] = None) -> go.Figure:
    """Mean ±1 SD against k, split by allocation arm. Average linkage only, for legibility."""
    sub = frame[frame["linkage"] == "average"]
    fig = go.Figure()
    for arm in ("random", "constrained-MacDonald"):
        cell = sub[sub["arm"] == arm]
        agg = cell.groupby("k")[value].agg(["mean", "std"]).reset_index()
        fig.add_trace(go.Scatter(
            x=agg["k"], y=agg["mean"], mode="lines+markers", name=arm,
            line=dict(color=COLOUR[arm], width=2), marker=dict(size=7),
            error_y=dict(type="data", array=agg["std"], visible=True, thickness=1)))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(dash="dash", color="rgba(128,128,128,0.8)"))
    ks = sorted(sub["k"].unique())
    fig.add_vrect(x0=HIGH_K, x1=max(ks), fillcolor="rgba(214,39,40,0.10)", line_width=0)
    fig.update_xaxes(title_text="clusters requested (k)", type="log", tickvals=ks,
                     range=[np.log10(min(ks) * 0.9), np.log10(max(ks) * 1.1)])
    fig.update_yaxes(title_text=ylab)
    return _style(fig, title)


# --------------------------------------------------------------------------- ground truth
def _scan_sd(run: Run, table: str, column: str) -> pd.DataFrame:
    """Mean ±1 SD across the stage-1 draws, from the raw per-draw table rather than the summary.

    The shipped ``*_summary.csv`` files carry a SEM. Every other figure in this report shows SD, so
    the spread means the same thing everywhere; the raw draws are on disk, so recompute rather than
    plot a different quantity here.
    """
    raw = run.stage1(table)
    return raw.groupby("ndim")[column].agg(["mean", "std"]).reset_index()


def gt_dimensionality_scan(run: Run) -> go.Figure:
    scan, cv = _scan_sd(run, "scan", "spearman"), _scan_sd(run, "cv", "spearman")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scan["ndim"], y=scan["mean"], mode="lines+markers", name="split-half",
        line=dict(color="#1f77b4", width=2), marker=dict(size=8),
        error_y=dict(type="data", array=scan["std"], visible=True, thickness=1)))
    fig.add_trace(go.Scatter(
        x=cv["ndim"], y=cv["mean"], mode="lines+markers", name="leave-k-out",
        line=dict(color="#2ca02c", width=2, dash="dot"), marker=dict(size=7),
        error_y=dict(type="data", array=cv["std"], visible=True, thickness=1)))
    fig.add_vline(x=8, line=dict(dash="dot", color="#d62728"),
                  annotation_text="d=8 used", annotation_position="top right")
    fig.update_xaxes(title_text="ground-truth dimensionality (d)")
    fig.update_yaxes(title_text="Spearman ρ")
    return _style(fig, "Split-half agreement peaks at d=3 and is flat across 3-5")


def gt_topk_by_ndim(run: Run) -> go.Figure:
    scan = _scan_sd(run, "scan", "topk_jaccard")
    fig = go.Figure(go.Scatter(x=scan["ndim"], y=scan["mean"], mode="lines+markers",
                               line=dict(color="#ff7f0e", width=2), marker=dict(size=8),
                               error_y=dict(type="data", array=scan["std"],
                                            visible=True, thickness=1)))
    fig.add_vline(x=8, line=dict(dash="dot", color="#d62728"),
                  annotation_text="d=8 used", annotation_position="top left")
    fig.update_xaxes(title_text="ground-truth dimensionality (d)")
    fig.update_yaxes(title_text=f"Jaccard overlap of the closest {TOP_FRAC_GT:.0%} of pairs "
                                f"(k = {K_GT:,})")
    return _style(fig, "Closest-pair agreement keeps rising to d=20, with no elbow to pick")


def gt_variance_spectrum(run: Run) -> go.Figure:
    """How the d=8 ground truth distributes its variance across its own axes."""
    coords = np.load(run.dir / "gt" / run.calibration()["gt_file"])
    centred = coords - coords.mean(axis=0)
    var = np.linalg.svd(centred, compute_uv=False) ** 2 / (len(coords) - 1)
    share = var / var.sum()
    axes = np.arange(1, len(share) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=axes, y=share, marker_color="#1f77b4", name="share of variance",
                         text=[f"{v:.0%}" for v in share], textposition="outside"))
    fig.add_trace(go.Scatter(x=axes, y=np.cumsum(share), mode="lines+markers", name="cumulative",
                             line=dict(color="#d62728", width=2), marker=dict(size=7), yaxis="y2"))
    fig.add_hline(y=1 / len(share), line=dict(dash="dot", color="rgba(128,128,128,0.8)"),
                  annotation_text="equal split across 8 axes", annotation_position="top right")
    fig.update_xaxes(title_text="principal axis of the ground-truth embedding", tickvals=axes)
    fig.update_layout(yaxis=dict(title="share of variance", range=[0, share.max() * 1.35]),
                      yaxis2=dict(title="cumulative", overlaying="y", side="right", range=[0, 1.05]))
    return _style(fig, "One dominant axis, then a near-flat tail: no elbow here either", height=400)


def gt_vs_noise_ceiling(run: Run) -> go.Figure:
    m = run.gt("gt_vs_raw")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["level_name"], y=m["spearman"], marker_color="#1f77b4",
                         name="BLUE: fitted embedding vs the ratings behind it (in-sample fit)"))
    fig.add_trace(go.Bar(x=m["level_name"], y=m["ceiling_full"], marker_color="#d62728",
                         name="RED: half the participants' ratings vs the other half "
                              "(noise ceiling)"))
    fig.update_xaxes(title_text="semantic relation, coarse to fine")
    fig.update_yaxes(title_text="Spearman ρ")
    return _style(fig, "Blue should not exceed red, and does", height=420, legend_rows=2)


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
    cal = run.calibration()
    labels = ["between-subject agreement<br>(mean pairwise Spearman ρ)",
              "within-subject test-retest<br>(Spearman ρ on repeats)"]
    empirical = [cal["empirical_agreement"], cal["target_test_retest"]]
    achieved = [cal["dispersion_achieved"], cal["achieved_tr_unscreened"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=empirical, name="empirical (pilot)", marker_color="#d62728",
                         text=[f"{v:.3f}" for v in empirical], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=achieved, name="achieved (simulated)", marker_color="#1f77b4",
                         text=[f"{v:.3f}" for v in achieved], textposition="outside"))
    fig.update_yaxes(title_text="Spearman ρ", range=[0, max(empirical + achieved) * 1.3])
    return _style(fig, "One target is hit; the other is undershot")


def realism_calibration_realised(run: Run) -> go.Figure:
    """Did the cohorts the sweep actually ran reproduce the calibrated reliability?

    ``realism_calibration`` shows the a-priori fit: what the calibration search settled on before
    any sweep cell was run. This one reads ``median_test_retest`` back off the 17,280 simulated
    cohorts, so a calibration that was fitted but not realised would show up here.
    """
    cov = run.table("coverage")
    cal = run.calibration()
    unscreened = cov[cov["screening_min_reliability"] < 0]
    fig = go.Figure()
    for arm in ("random", "constrained-MacDonald"):
        sub = unscreened[unscreened["arm"] == arm]
        fig.add_trace(go.Box(y=sub["median_test_retest"], name=arm, marker_color=COLOUR[arm],
                             boxpoints=False, showlegend=False))
    fig.add_hline(y=cal["achieved_tr_unscreened"], line=dict(dash="dash", color="#1f77b4"),
                  annotation_text=f"calibration fit ({cal['achieved_tr_unscreened']:.3f})",
                  annotation_position="bottom right")
    fig.add_hline(y=cal["target_test_retest"], line=dict(dash="dot", color="#d62728"),
                  annotation_text=f"pilot target ({cal['target_test_retest']:.3f})",
                  annotation_position="top right")
    fig.update_yaxes(title_text="cohort median test-retest Spearman ρ")
    fig.update_xaxes(title_text="allocation arm")
    return _style(fig, "The cohorts realise the calibrated value, which is itself short of the "
                       "pilot", height=400)


def realism_null_distance(run: Run) -> go.Figure:
    """Simulated and pilot distances against random placement on the same canvas.

    The null is ``analysis/pilot/simulate_null_distances.py``: 20 points dropped uniformly on the
    unit square, distances divided by the diagonal. Under task-v5 the simulated distances are
    divided by the canvas diagonal too, so all three live on the same [0, 1] axis with no rescaling.
    """
    s = run.table("null_distances").set_index("source")
    null = float(s.loc["random placement (null)", "mean"])
    order = ["pilot participants", "random placement (null)", "simulated participants"]
    colour = {"random placement (null)": "#7f7f7f", "simulated participants": COLOUR["sim"],
              "pilot participants": COLOUR["pilot"]}
    fig = go.Figure()
    for name in order:
        row = s.loc[name]
        gap = 100 * (row["mean"] - null) / null
        tag = "chance" if name.startswith("random") else f"{gap:+.0f}% vs chance"
        fig.add_trace(go.Bar(
            x=[row["mean"]], y=[name], orientation="h", marker_color=colour[name],
            showlegend=False,
            error_x=dict(type="data", array=[row["sd"]], visible=True, thickness=1),
            text=[f"  {row['mean']:.3f} ± {row['sd']:.3f}   ({tag})"], textposition="outside"))
    fig.add_vline(x=null, line=dict(dash="dash", color="#7f7f7f"))
    fig.update_xaxes(title_text="mean pairwise distance, in canvas-diagonal units", range=[0, 0.78])
    return _style(fig, "The pilot clusters items; the simulation spreads them past chance",
                  height=300)


def realism_noise_vs_distance(run: Run) -> go.Figure:
    """The binned-RMSE curve in native canvas-diagonal units.

    Both sources are already normalised by their own canvas diagonal, so no rescaling is needed and
    the axis reads directly in the units the task records. The simulated band is ±1 SD across 20
    independent cohorts; the pilot is a single fixed sample, so it has no band.
    """
    curves = run.table("noise_vs_distance_native")
    fig = go.Figure()
    for src, label in (("pilot", "pilot participants (one fixed sample)"),
                       ("sim", "simulated participants (20 cohorts)")):
        sub = curves[curves["source"] == src]
        err = (dict(type="data", array=sub["sd_rmse"], visible=True, thickness=1)
               if src == "sim" else None)
        fig.add_trace(go.Scatter(x=sub["mean_pair_distance"], y=sub["rmse"], mode="lines+markers",
                                 name=label, line=dict(color=COLOUR[src], width=2), error_y=err))
    fig.update_xaxes(title_text="mean of the two placements (canvas diagonals)", range=[0, 1])
    fig.update_yaxes(title_text="RMSE between the two placements (canvas diagonals)")
    return _style(fig, "Both curves rise then fall: the inverted U the model was not fitted to")


def realism_semantic_gradient(run: Run) -> go.Figure:
    g = run.table("validity_gradient")
    fig = go.Figure()
    pilot = g[g["arm"] == "random"]
    fig.add_trace(go.Scatter(
        x=pilot["level_name"], y=pilot["mean_distance_pilot"] / pilot["mean_distance_pilot"].iloc[0],
        mode="lines+markers", name="pilot participants",
        line=dict(color=COLOUR["pilot"], width=2), marker=dict(size=8)))
    for arm, dash in (("random", "dot"), ("designed", "solid")):
        sub = g[g["arm"] == arm]
        if sub.empty:
            continue
        label = "simulated, " + ("random" if arm == "random" else "constrained-MacDonald")
        colour = COLOUR["random"] if arm == "random" else COLOUR["constrained-MacDonald"]
        fig.add_trace(go.Scatter(
            x=sub["level_name"], y=sub["mean_distance_sim"] / sub["mean_distance_sim"].iloc[0],
            mode="lines+markers", name=label, line=dict(color=colour, width=2, dash=dash),
            marker=dict(size=7)))
    fig.update_xaxes(title_text="semantic relation, coarse to fine")
    fig.update_yaxes(title_text="mean distance / unrelated-pair distance")
    return _style(fig, "Both arms order the levels correctly and both under-separate the fine end")


# --------------------------------------------------------------------------- coverage
def coverage_by_n(run: Run) -> go.Figure:
    cov = run.table("coverage")
    fig = _by_arm(cov, "pair_coverage",
                  "Share of the 262,450 image pairs judged by anyone", "pairs covered (%)")
    fig.add_hline(y=100, line=dict(dash="dot", color="rgba(128,128,128,0.6)"))
    means = cov.groupby(["num_subjects", "arm"])["pair_coverage"].mean().unstack()
    for n in means.index:
        rand, des = means.loc[n, "random"], means.loc[n, "constrained-MacDonald"]
        fig.add_annotation(x=np.log10(n), y=des, text=f"<b>{100 * (des - rand) / rand:+.1f}%</b>",
                           showarrow=False, yshift=26, font=dict(size=11, color="#1f77b4"))
    return fig


def allocation_balance(run: Run) -> go.Figure:
    d = run.table("design_only")
    g = d.groupby(["num_subjects", "arm"])["reps_per_image_sd"].agg(["mean", "std"])
    fig = go.Figure()
    for arm, label, colour in (("random", "random", COLOUR["random"]),
                               ("designed", "constrained MacDonald",
                                COLOUR["constrained-MacDonald"]),
                               ("designed_unconstrained", "unconstrained MacDonald", "#9467bd")):
        if arm not in d["arm"].unique():
            continue
        sub = g.xs(arm, level="arm")
        fig.add_trace(go.Bar(x=[str(int(n)) for n in sub.index], y=sub["mean"], name=label,
                             marker_color=colour,
                             error_y=dict(type="data", array=sub["std"], visible=True,
                                          thickness=1)))
    fig.update_xaxes(title_text="participants (N)")
    fig.update_yaxes(title_text="SD of appearances per image")
    return _style(fig, "How evenly the 725 images are used (lower is more balanced)")


def constraint_cost(run: Run) -> go.Figure:
    d = run.table("design_only")
    fig = go.Figure()
    for arm, label, dash, colour in (
            ("random", "random", "solid", COLOUR["random"]),
            ("designed", "constrained MacDonald", "solid", COLOUR["constrained-MacDonald"]),
            ("designed_unconstrained", "unconstrained MacDonald", "dot", "#9467bd")):
        if arm not in d["arm"].unique():
            continue
        agg = d[d["arm"] == arm].groupby("num_subjects")["frac_pairs_covered"].agg(
            ["mean", "std"]).reset_index()
        fig.add_trace(go.Scatter(x=agg["num_subjects"], y=100 * agg["mean"], mode="lines+markers",
                                 name=label, line=dict(width=2, dash=dash, color=colour),
                                 error_y=dict(type="data", array=100 * agg["std"], visible=True,
                                              thickness=1)))
    fig.update_xaxes(title_text="participants (N)", type="log",
                     tickvals=sorted(d["num_subjects"].unique()))
    fig.update_yaxes(title_text="pairs covered (%)")
    return _style(fig, "Session-disjointness costs almost nothing in coverage")


def mds_convergence(run: Run) -> go.Figure:
    m = run.meta()
    ok, hit = m[m["status"] == "success"], m[m["status"] == "max_iters"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ok["stress"], nbinsx=60, marker_color="#1f77b4",
                               name=f"converged (n={len(ok):,})"))
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
                   "Agreement between two cohorts' pooled ratings, before MDS", "Spearman ρ")


def stability_embedding(run: Run) -> go.Figure:
    return _by_arm(run.table("embedding_stability"), "mean_spearman",
                   "Agreement between two cohorts' recovered embeddings",
                   "Spearman ρ between recovered distances")


def stability_procrustes(run: Run) -> go.Figure:
    return _by_arm(run.table("embedding_generalizability"), "mean_procrustes_m2",
                   "Procrustes m² between two cohorts' embeddings (lower is better)",
                   "Procrustes m²")


# --------------------------------------------------------------------------- the paired test
AXES = ["canvas_softness", "screening_min_reliability", "perspective_dispersion"]
_TESTS = [("stability", "spearman", "raw ratings (Spearman ρ)", True),
          ("embedding_stability", "mean_spearman", "embeddings (Spearman ρ)", True),
          ("embedding_generalizability", "mean_procrustes_m2", "embeddings (Procrustes m²)", False),
          ("topk_jaccard", "mean_jaccard", "closest pairs (top-k Jaccard)", True),
          ("recovery_vs_gt", "mean_auc", "recovery (AUC)", True),
          ("recovery_vs_gt", "mean_recall", "recovery (recall@k)", True)]


def _paired_at_n(run: Run, n: int = 50) -> pd.DataFrame:
    """Paired comparison of the two arms at N, matched within every swept setting.

    Pairing matters: both arms are evaluated at identical settings, so the between-setting variance
    is removed rather than being counted as noise. Differences are oriented so positive favours the
    designed arm.
    """
    rows = []
    for table, col, label, higher_better in _TESTS:
        df = run.table(table)
        df = df[df["num_subjects"] == n]
        keys = [c for c in AXES + ["ndim", "top_frac", "rep", "rep_i", "rep_j"] if c in df.columns]
        wide = df.groupby(keys + ["arm"])[col].mean().unstack("arm").dropna()
        diff = (wide["constrained-MacDonald"] - wide["random"]).to_numpy()
        if not higher_better:
            diff = -diff
        t, p = stats.ttest_1samp(diff, 0.0)
        rows.append({
            "measure": label, "settings": diff.size,
            "random": wide["random"].mean(), "designed": wide["constrained-MacDonald"].mean(),
            "difference": diff.mean(),
            "dz": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan,
            "p": p, "favours designed": f"{100 * (diff > 0).mean():.0f}%"})
    return pd.DataFrame(rows)


def stability_significance(run: Run) -> go.Figure:
    d = _paired_at_n(run, 50)

    def fmt_p(p):
        return "&lt; 1e-15" if p < 1e-15 else f"{p:.1e}"

    header = ["measure", "random", "designed", "difference", "Cohen's d<sub>z</sub>", "p",
              "settings favouring<br>the designed arm"]
    fig = go.Figure(go.Table(
        columnwidth=[210, 85, 85, 95, 105, 85, 155],
        header=dict(values=header, fill_color="rgba(31,119,180,0.15)", align="left",
                    height=44, font=dict(size=12)),
        cells=dict(align="left", height=34, font=dict(size=12), values=[
            d["measure"], d["random"].round(4), d["designed"].round(4),
            d["difference"].map(lambda v: f"{v:+.4f}"), d["dz"].round(2),
            d["p"].map(fmt_p), d["favours designed"]])))
    fig.update_layout(height=44 + 34 * len(d) + 60, margin=dict(l=0, r=0, t=52, b=8),
                      paper_bgcolor=PLOT_BG,
                      title=dict(text="Paired comparison at N=50, matched within every setting",
                                 font=dict(size=14)))
    return fig


# --------------------------------------------------------------------------- closest pairs
def _by_arm_panels_over_k(frame: pd.DataFrame, value: str, title: str, ylab: str,
                          fracs: Sequence[float], hline: Optional[float] = None) -> go.Figure:
    """One panel per top-k cut, so the axis label can name an actual k instead of hiding it.

    A tighter cut is a harder question: the top 1% of pairs is the neighbourhood the study is really
    about, and the top 10% is a much more forgiving target.
    """
    fig = make_subplots(rows=1, cols=len(fracs), shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[k_label(f) for f in fracs])
    for col, frac in enumerate(fracs, start=1):
        cut = frame[np.isclose(frame["top_frac"], frac)]
        for arm in ("random", "constrained-MacDonald"):
            sub = cut[cut["arm"] == arm]
            agg = sub.groupby("num_subjects")[value].agg(["mean", "std"]).reset_index()
            fig.add_trace(go.Scatter(
                x=agg["num_subjects"], y=agg["mean"], mode="lines+markers", name=arm,
                legendgroup=arm, showlegend=(col == 1),
                line=dict(color=COLOUR[arm], width=2), marker=dict(size=7),
                error_y=dict(type="data", array=agg["std"], visible=True, thickness=1)),
                row=1, col=col)
        if hline is not None:
            fig.add_hline(y=hline, line=dict(dash="dot", color="rgba(128,128,128,0.7)"),
                          row=1, col=col)
        fig.update_xaxes(title_text="participants (N)", type="log",
                         tickvals=sorted(frame["num_subjects"].unique()), row=1, col=col)
    fig.update_yaxes(title_text=ylab, row=1, col=1)
    fig = _style(fig, title, height=420)
    fig.update_annotations(font_size=11)
    return fig


def topk_between_cohorts(run: Run) -> go.Figure:
    return _by_arm_panels_over_k(
        run.table("topk_jaccard"), "mean_jaccard",
        "Do two cohorts agree on which pairs are closest?",
        "Jaccard overlap", fracs=(0.05, 0.1, 0.25))


def recovery_recall(run: Run) -> go.Figure:
    return _by_arm_panels_over_k(
        run.table("recovery_vs_gt"), "mean_recall",
        "Recall of the ground truth's closest pairs",
        "recall@k", fracs=(0.01, 0.05, 0.1))


def recovery_auc(run: Run) -> go.Figure:
    return _by_arm(run.table("recovery_vs_gt"), "mean_auc",
                   "Separating truly-close pairs from far ones", "AUC", hline=0.5)


def recovery_dprime(run: Run) -> go.Figure:
    return _by_arm(run.table("recovery_vs_gt"), "mean_dprime",
                   "Discriminability of the closest pairs", "d′")


# --------------------------------------------------------------------------- clusters
def cluster_vi_by_k(run: Run) -> go.Figure:
    return _by_arm_and_k(run.table("cluster_agreement"), "mean_vi_norm",
                         "Partition agreement against granularity, by allocation arm",
                         "normalised VI (lower is more agreement)")


def cluster_ari_by_k(run: Run) -> go.Figure:
    return _by_arm_and_k(run.table("cluster_agreement"), "mean_ari",
                         "Adjusted Rand index against granularity, by allocation arm",
                         "ARI (higher is more agreement)")


def cluster_silhouette_by_k(run: Run) -> go.Figure:
    return _by_arm_and_k(run.table("cluster_agreement"), "mean_sil_cross",
                         "Do the clusters still separate in another cohort's geometry?",
                         "cross-cohort silhouette", hline=0.0)


def cluster_dendrogram_agreement(run: Run) -> go.Figure:
    d = run.table("dendrogram_agreement")
    fig = go.Figure()
    for value, label, colour in (("mean_baker_gamma", "Baker's γ", "#1f77b4"),
                                 ("mean_cophenetic_fidelity", "cophenetic correlation", "#2ca02c")):
        agg = d.groupby("linkage")[value].agg(["mean", "std"]).reset_index()
        fig.add_trace(go.Bar(x=agg["linkage"], y=agg["mean"], name=label, marker_color=colour,
                             error_y=dict(type="data", array=agg["std"], visible=True,
                                          thickness=1)))
    fig.update_xaxes(title_text="linkage")
    fig.update_yaxes(title_text="correlation")
    return _style(fig, "Whole-tree agreement, independent of any cut")


def cluster_density(run: Run) -> go.Figure:
    """Noise fraction and cohort agreement on it, split by arm and sample size."""
    d = run.table("density_agreement")
    fig = go.Figure()
    for arm in ("random", "constrained-MacDonald"):
        for value, dash, label in (("mean_frac_noise", "solid", "left unclustered"),
                                   ("mean_noise_kappa", "dot", "Cohen's κ on which")):
            sub = d[d["arm"] == arm]
            agg = sub.groupby("min_cluster_size")[value].agg(["mean", "std"]).reset_index()
            fig.add_trace(go.Scatter(
                x=agg["min_cluster_size"], y=agg["mean"], mode="lines+markers",
                name=f"{label}, {arm}",
                line=dict(color=COLOUR[arm], width=2, dash=dash), marker=dict(size=7),
                error_y=dict(type="data", array=agg["std"], visible=True, thickness=1)))
    fig.update_xaxes(title_text="HDBSCAN min_cluster_size")
    fig.update_yaxes(title_text="proportion")
    return _style(fig, "Most images belong to no cluster, and the two arms behave alike",
                  height=420, legend_rows=2)


def cluster_density_by_n(run: Run) -> go.Figure:
    d = run.table("density_agreement")
    d = d[d["min_cluster_size"] == 5]
    fig = _by_arm(d, "mean_frac_noise",
                  "Images left unclustered at min_cluster_size=5, against sample size",
                  "share left unclustered")
    return fig


# --------------------------------------------------------------------------- screening
def screening_cost(run: Run) -> go.Figure:
    cov = run.table("coverage")
    g = cov.groupby("screening_min_reliability").agg(
        pass_rate=("screening_pass_rate", "mean"),
        pass_sd=("screening_pass_rate", "std"),
        retained_reliability=("median_test_retest", "mean"),
        retained_sd=("median_test_retest", "std")).reset_index()
    g["candidates"] = 1 / g["pass_rate"]
    labels = [SCREEN_LABEL[t] for t in g["screening_min_reliability"]]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("What it costs: candidates per retained subject",
                                        "What it buys: test-retest ρ of the retained cohort"))
    fig.add_trace(go.Bar(x=labels, y=g["candidates"], marker_color="#d62728", showlegend=False,
                         text=[f"{v:.2f}×" for v in g["candidates"]], textposition="outside"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=g["retained_reliability"], marker_color="#1f77b4",
                         showlegend=False,
                         error_y=dict(type="data", array=g["retained_sd"], visible=True,
                                      thickness=1),
                         text=[f"{v:.3f}" for v in g["retained_reliability"]],
                         textposition="outside"), row=2, col=1)
    fig.update_yaxes(title_text="candidates", range=[0, g["candidates"].max() * 1.35], row=1, col=1)
    fig.update_yaxes(title_text="test-retest ρ", range=[0, g["retained_reliability"].max() * 1.45],
                     row=2, col=1)
    fig.update_xaxes(title_text="screening threshold", row=2, col=1)
    fig = _style(fig, "Recruitment cost rises far faster than the quality it buys", height=560)
    fig.update_annotations(font_size=12)
    return fig


def screening_pays_off(run: Run) -> go.Figure:
    """Does a better-screened cohort produce a better map, at fixed N?

    The screening threshold is the one swept axis that changes participant *quality* rather than
    quantity, so it is the only handle this sweep has on the question. Every panel holds N=50 and
    averages over the remaining swept settings, so the only thing moving is who was retained.
    """
    panels = [("embedding_stability", "mean_spearman", "cohort agreement<br>(Spearman ρ)"),
              ("recovery_vs_gt", "mean_auc", "recovery of the GT<br>(AUC)"),
              ("topk_jaccard", "mean_jaccard", "closest-pair agreement<br>(Jaccard)")]
    cov = run.table("coverage")
    quality = (cov[cov["num_subjects"] == 50].groupby("screening_min_reliability")
               ["median_test_retest"].mean())
    fig = make_subplots(rows=1, cols=3, subplot_titles=[p[2] for p in panels],
                        horizontal_spacing=0.09)
    for col, (table, value, _) in enumerate(panels, start=1):
        df = run.table(table)
        df = df[df["num_subjects"] == 50]
        for arm in ("random", "constrained-MacDonald"):
            sub = df[df["arm"] == arm]
            agg = sub.groupby("screening_min_reliability")[value].agg(["mean", "std"]).reset_index()
            fig.add_trace(go.Scatter(
                x=[quality[t] for t in agg["screening_min_reliability"]], y=agg["mean"],
                mode="lines+markers", name=arm, legendgroup=arm, showlegend=(col == 1),
                line=dict(color=COLOUR[arm], width=2), marker=dict(size=8),
                error_y=dict(type="data", array=agg["std"], visible=True, thickness=1)),
                row=1, col=col)
        fig.update_xaxes(title_text="retained cohort's test-retest ρ", row=1, col=col)
    fig = _style(fig, "At fixed N=50, a cleaner cohort is worth a great deal", height=400)
    fig.update_annotations(font_size=12)
    return fig


SCREEN_COLOUR = {-1.0: "#888888", 0.0: "#1f77b4", 0.1: "#2ca02c", 0.2: "#d62728"}


def screening_per_recruit(run: Run) -> go.Figure:
    """The same gain, charged against the recruitment it costs rather than the cohort it leaves.

    Screening raises quality per *retained* participant, but every rejected candidate was paid for.
    Plotting against ``N / pass_rate`` puts all four thresholds on one budget axis, which is the
    axis the decision is actually made on.
    """
    es, cov = run.table("embedding_stability"), run.table("coverage")
    pass_rate = cov.groupby("screening_min_reliability")["screening_pass_rate"].mean()
    g = (es.groupby(["screening_min_reliability", "num_subjects"])["mean_spearman"]
         .agg(["mean", "std"]).reset_index())
    g["recruited"] = g["num_subjects"] / g["screening_min_reliability"].map(pass_rate)
    fig = go.Figure()
    for threshold, sub in g.groupby("screening_min_reliability"):
        sub = sub.sort_values("recruited")
        fig.add_trace(go.Scatter(
            x=sub["recruited"], y=sub["mean"], mode="lines+markers",
            name=SCREEN_LABEL[threshold], line=dict(color=SCREEN_COLOUR[threshold], width=2),
            marker=dict(size=8),
            error_y=dict(type="data", array=sub["std"], visible=True, thickness=1)))
    fig.update_xaxes(title_text="participants <b>recruited and paid</b> (N ÷ pass rate)", type="log")
    fig.update_yaxes(title_text="Spearman ρ between cohorts")
    return _style(fig, "Charged against recruitment, the four thresholds nearly coincide",
                  height=420)


FIGURES: Dict[str, Fig] = {
    "gt_dimensionality_scan": Fig(
        gt_dimensionality_scan,
        "Error bars are ±1 SD across 50 random half-splits of the pilot."),
    "gt_topk_by_ndim": Fig(
        gt_topk_by_ndim,
        f"k is a count of pairs, not of images: the closest {TOP_FRAC_GT:.0%} of all "
        f"{N_PAIRS:,} pairs, so k = {K_GT:,}. Error bars are ±1 SD across 50 random half-splits "
        f"of the pilot."),
    "gt_variance_spectrum": Fig(
        gt_variance_spectrum,
        "Principal-axis variance of the fitted d=8 coordinates, computed once on the fixed ground "
        "truth. Read it as a description of the solution, not as evidence for d=8: SMACOF "
        "minimises stress using every dimension it is given, so it fills them all."),
    "gt_vs_noise_ceiling": Fig(
        gt_vs_noise_ceiling,
        "Both bars are computed on the 41 pre-SHINE pilot participants. RED is a split-half "
        "reliability: the participants are split at random into two halves, each half's mean rating "
        "per pair is computed, the two are correlated, and the result is Spearman-Brown corrected "
        "to the full sample. It is what these ratings can agree with themselves on, and therefore "
        "the most any model of them could legitimately reach."),
    "gt_semantic_gradient": Fig(
        gt_semantic_gradient,
        "Computed once on the fixed ground truth, so there is no spread to show."),
    "realism_calibration": Fig(
        realism_calibration,
        "Point estimates from the calibration fit, which is an a-priori search run before the "
        "sweep; the empirical bars are medians over the pilot participants who contribute to each "
        "observable (n=47 and n=22 respectively). The next figure checks the fit against the "
        "cohorts actually simulated."),
    "realism_calibration_realised": Fig(
        realism_calibration_realised,
        "Boxes span the quartiles of cohort-median test-retest ρ across all 2,160 unscreened "
        "cohorts per arm; whiskers are 1.5 IQR. This is measured output, not a fitted value."),
    "realism_noise_vs_distance": Fig(
        realism_noise_vs_distance,
        "Both axes are in canvas-diagonal units, the same normalisation the deployed task "
        "records, so no rescaling is applied. Error bars are ±1 SD across 20 independently "
        "simulated cohorts; the pilot is one fixed sample and has none. Bins hold equal counts, so "
        "the points are unevenly spaced along x."),
    "realism_null_distance": Fig(
        realism_null_distance,
        "Error bars are ±1 SD across all pairs in each source. The null is 2,000 trials of 20 "
        "points dropped uniformly on the canvas, from the same script the pilot analysis uses "
        "(analysis/pilot/simulate_null_distances.py)."),
    "realism_semantic_gradient": Fig(
        realism_semantic_gradient,
        "One simulated cohort per arm at N=500, so there is no spread to show. The caveat "
        "immediately below the figure explains what that does and does not license."),
    "coverage_by_n": Fig(
        coverage_by_n,
        "Error bars are ±1 SD across the 360 simulated cohorts at each point (10 repetitions × 36 "
        "swept settings). Blue labels give the relative gain over random."),
    "allocation_balance": Fig(
        allocation_balance,
        "Error bars are ±1 SD across 20 independently generated designs per point. Pure "
        "combinatorics: no simulated participants are involved."),
    "constraint_cost": Fig(
        constraint_cost,
        "Error bars are ±1 SD across 20 independently generated designs per point."),
    "mds_convergence": Fig(
        mds_convergence,
        "All 17,729 fits in the store, including the 449 duplicates a resumed run appended."),
    "stability_raw": Fig(stability_raw, SD_NOTE),
    "stability_embedding": Fig(stability_embedding, SD_NOTE),
    "stability_procrustes": Fig(stability_procrustes, SD_NOTE),
    "stability_significance": Fig(
        stability_significance,
        "Paired t-test on the difference between arms, matched within each of the swept settings "
        "at N=50. p-values are descriptive: the settings form a designed grid rather than a random "
        "sample, so the effect size and the proportion of settings favouring one arm carry more "
        "weight than the p-value."),
    "topk_between_cohorts": Fig(
        topk_between_cohorts,
        "k is a count of pairs, not of images, and each panel is a different cut. " + SD_NOTE),
    "recovery_recall": Fig(
        recovery_recall,
        "k is a count of pairs, not of images. The left panel is the hardest question the study "
        "actually asks: of the 2,624 pairs the ground truth calls closest, how many appear in the "
        "recovered top 2,624. " + SD_NOTE),
    "recovery_auc": Fig(recovery_auc, SD_NOTE),
    "recovery_dprime": Fig(recovery_dprime, SD_NOTE),
    "cluster_vi_by_k": Fig(
        cluster_vi_by_k,
        "Average linkage only, for legibility; ward and complete behave the same way at a lower "
        "level. Error bars are ±1 SD across every cohort pair and swept setting. The shaded band "
        "marks k ≥ 150, discussed in the text."),
    "cluster_ari_by_k": Fig(
        cluster_ari_by_k,
        "Average linkage only. Error bars are ±1 SD across every cohort pair and swept setting."),
    "cluster_silhouette_by_k": Fig(
        cluster_silhouette_by_k,
        "Average linkage only. Error bars are ±1 SD across every cohort pair and swept setting."),
    "cluster_dendrogram_agreement": Fig(
        cluster_dendrogram_agreement,
        "Error bars are ±1 SD across every cohort pair and swept setting."),
    "cluster_density": Fig(
        cluster_density,
        "Error bars are ±1 SD across every cohort pair and swept setting, pooled over sample "
        "sizes."),
    "cluster_density_by_n": Fig(
        cluster_density_by_n,
        "At min_cluster_size=5. " + SD_NOTE),
    "screening_cost": Fig(
        screening_cost,
        "Bars are means over every configuration at that threshold; error bars are ±1 SD of the "
        "retained cohorts' median test-retest reliability. Note the shared x-axis: the two panels "
        "are the two halves of one trade."),
    "screening_pays_off": Fig(
        screening_pays_off,
        "N=50 throughout, so only participant quality varies. x is the retained cohort's measured "
        "test-retest ρ, which is what the threshold buys; error bars are ±1 SD across every cohort "
        "and remaining swept setting."),
    "screening_per_recruit": Fig(
        screening_per_recruit,
        "The same four thresholds, with each cohort's x moved from N to N ÷ pass rate: the number "
        "of people who had to be recruited and paid to retain it. Error bars are ±1 SD across every "
        "cohort and swept setting. Both axes are logarithmic."),
}
