"""Every figure in the v6 report, one entry per named placeholder.

v6 is a DECISION run, not a survey, so its figures answer one question each: which of three ways to
finish the study is best, and can RQ2 be answered at either N. Nothing here restates v5 - where a
quantity is unchanged from v5 the prose cites it rather than re-plotting it.

Two conventions differ from v5's figures and both are deliberate.

**Drift is a swept axis, never averaged over.** The gate passes at both 1.0 and 1.1, so the run
carries both, and collapsing them would hide the one thing the sweep exists to show: whether the
three options separate the same way at each end. Every figure that can show drift shows it.

**Cells are named, not encoded.** Three cells is few enough to label in full ("N=50, rho>0"), and a
reader deciding how to spend several hundred credits should never have to decode a legend.

Register a figure in :data:`FIGURES`. ``assemble.py`` fails if the source references a name that is
not registered, or if a registered name is never used, so prose and figures cannot drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PLOT_BG = "rgba(0,0,0,0)"

# The three options, in the order they are always presented: current, then the two ways to spend.
#
# Everything downstream keys off the STRING label, never the (N, threshold) tuple. A tuple used as a
# pandas label is ambiguous - `frame.loc[(50, 0.0), "mean"]` is read as multi-axis indexing, one
# selector per axis, rather than as a single key - and it fails with a bare AssertionError from deep
# inside .loc, which says nothing about the cause.
CELLS = [(50, 0.0), (50, 0.1), (75, 0.0)]
CELL_LABEL = {(50, 0.0): "N=50, ρ>0", (50, 0.1): "N=50, ρ>0.1", (75, 0.0): "N=75, ρ>0"}
CELL_ORDER = [CELL_LABEL[c] for c in CELLS]
CELL_COLOUR = {CELL_LABEL[(50, 0.0)]: "#888888", CELL_LABEL[(50, 0.1)]: "#1f77b4",
               CELL_LABEL[(75, 0.0)]: "#2ca02c"}
# Credits to finish from the position the study paused at (~80 retained, 39/39 per SHINE cohort).
CELL_COST = {CELL_LABEL[(50, 0.0)]: 286, CELL_LABEL[(50, 0.1)]: 636, CELL_LABEL[(75, 0.0)]: 683}

DRIFT_DASH = {1.0: "solid", 1.1: "dot"}
DRIFT_LABEL = {1.0: "no drift", 1.1: "drift 1.1"}

SD_NOTE = "Error bars are ±1 SD across every simulated cohort and swept setting in that cell."
DRIFT_NOTE = ("Both within-session drift values are shown because the calibration gate passes at "
              "each; solid is no drift, dotted is drift 1.1.")


class Fig(NamedTuple):
    fn: Callable[["Run"], go.Figure]
    caption: str


class Run:
    """The run's tables, loaded once and shared by every figure.

    Deliberately tolerant: this report is built while the run is still producing output, so a
    missing table yields a placeholder panel rather than a traceback. A figure that silently
    invented data would be far worse than one that says it has none yet.
    """

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    def table(self, name: str, subdir: str = "out") -> Optional[pd.DataFrame]:
        key = f"{subdir}/{name}"
        if key not in self._cache:
            path = self.dir / subdir / f"{name}.csv"
            self._cache[key] = pd.read_csv(path) if path.is_file() else None
        return self._cache[key]

    def json(self, name: str, subdir: str) -> Optional[dict]:
        path = self.dir / subdir / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def cells(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Restrict to the three compared cells, labelling each with its display string."""
        keys = zip(frame["num_subjects"].astype(int),
                   frame["screening_min_reliability"].astype(float))
        frame = frame.assign(cell=[CELL_LABEL.get(k) for k in keys])
        return frame[frame["cell"].notna()]


def _pending(message: str) -> go.Figure:
    """A visible, honest placeholder for a table the run has not produced yet."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=13, color="#888"),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(height=200, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _drift_values(frame: pd.DataFrame) -> list:
    if "within_session_drift" not in frame.columns:
        return [None]
    return sorted(frame["within_session_drift"].unique())


# --------------------------------------------------------------------------- the gate
def validation_gate(run: Run) -> go.Figure:
    """Observed against simulated, for the six quantities production already shows."""
    gate = run.json("validation_gate", "calibration")
    if gate is None:
        return _pending("calibration/validation_gate.json not found - run the gate first")
    names = list(gate["targets"])
    observed = [gate["targets"][n] for n in names]
    simulated = [gate["simulated"][n] for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="observed (production)", x=names, y=observed,
                         marker_color="#d62728", opacity=0.85))
    fig.add_trace(go.Bar(name="simulated", x=names, y=simulated, marker_color="#1f77b4"))
    fig.update_layout(barmode="group", height=380, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      yaxis_title="value", legend=dict(orientation="h", y=1.12),
                      margin=dict(l=60, r=20, t=40, b=90))
    return fig


# --------------------------------------------------------------------------- the comparison
# The pipeline's tables arrive PRE-AGGREGATED over reps: one row per (params, ndim) carrying
# `mean_<x>` and `sem_<x>`, not one row per cohort. `coverage` is the exception - it is per-rep and
# has no `ndim` column at all. Both shapes go through `_cell_metric`.
HEADLINE_NDIM = 8          # the dimensionality the GT was built at; see gt/gt_v6_decision.json
HEADLINE_TOP_FRAC = 0.05   # gt_construction.DEFAULT_TOP_FRAC


def _cell_metric(run: Run, table: str, column: str, title: str, *,
                 top_frac: Optional[float] = None) -> go.Figure:
    frame = run.table(table)
    if frame is None or column not in frame.columns:
        return _pending(f"out/{table}.csv has no column '{column}' - the run did not produce it")
    # Fix ndim and top_frac rather than averaging over them. Averaging a metric across
    # dimensionalities mixes different fits, and averaging across k answers no question anyone
    # asked - that mistake turned an ARI of 0.549 into 0.194 earlier in this project.
    if "ndim" in frame.columns:
        frame = frame[frame["ndim"] == HEADLINE_NDIM]
    if top_frac is not None and "top_frac" in frame.columns:
        frame = frame[np.isclose(frame["top_frac"], top_frac)]
    frame = run.cells(frame)
    if frame.empty:
        return _pending(f"out/{table}.csv has no rows for the three compared cells")

    fig = go.Figure()
    drifts = _drift_values(frame)
    for drift in drifts:
        sub = frame if drift is None else frame[frame["within_session_drift"] == drift]
        # SD across the swept settings (dispersion x softness), not the SEM the table carries:
        # the question is how much the answer moves across the sensitivity arm, not how precisely
        # each cell's mean is known.
        grouped = sub.groupby("cell")[column].agg(["mean", "std"])
        present = [c for c in CELL_ORDER if c in grouped.index]
        fig.add_trace(go.Bar(
            name=DRIFT_LABEL.get(drift, "all"), x=present,
            y=[grouped["mean"][c] for c in present],
            error_y=dict(type="data", array=[grouped["std"][c] for c in present], visible=True),
            marker_color=[CELL_COLOUR[c] for c in present],
            opacity=1.0 if drift in (None, 1.0) else 0.55))
    fig.update_layout(barmode="group", height=380, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      yaxis_title=title, showlegend=len(drifts) > 1,
                      legend=dict(orientation="h", y=1.12), margin=dict(l=60, r=20, t=40, b=50))
    return fig


def recovery_by_cell(run: Run) -> go.Figure:
    return _cell_metric(run, "recovery_vs_gt", "mean_recall",
                        f"recall of the true closest {HEADLINE_TOP_FRAC:.0%} of pairs",
                        top_frac=HEADLINE_TOP_FRAC)


def stability_by_cell(run: Run) -> go.Figure:
    return _cell_metric(run, "embedding_stability", "mean_spearman", "between-cohort Spearman")


def jaccard_by_cell(run: Run) -> go.Figure:
    return _cell_metric(run, "topk_jaccard", "mean_jaccard",
                        f"top-{HEADLINE_TOP_FRAC:.0%} closest-pair Jaccard",
                        top_frac=HEADLINE_TOP_FRAC)


def coverage_by_cell(run: Run) -> go.Figure:
    # `pair_coverage` is a percentage in this table (48.1, not 0.481), so the axis says so.
    return _cell_metric(run, "coverage", "pair_coverage", "% of image pairs observed")


def cost_per_quality(run: Run) -> go.Figure:
    """The decision, stated as what each option costs against what it buys.

    The credits are the marginal cost of FINISHING from where collection paused, not the cost of the
    whole study, because that is the only part that can still be spent differently.
    """
    frame = run.table("recovery_vs_gt")
    if frame is None:
        return _pending("out/recovery_vs_gt.csv not found")
    frame = frame[(frame["ndim"] == HEADLINE_NDIM)
                  & np.isclose(frame["top_frac"], HEADLINE_TOP_FRAC)]
    frame = run.cells(frame)
    if frame.empty:
        return _pending("out/recovery_vs_gt.csv has no rows for the three compared cells")
    fig = go.Figure()
    for drift in _drift_values(frame):
        sub = frame if drift is None else frame[frame["within_session_drift"] == drift]
        grouped = sub.groupby("cell")["mean_recall"].mean()
        present = [c for c in CELL_ORDER if c in grouped.index]
        fig.add_trace(go.Scatter(
            x=[CELL_COST[c] for c in present], y=[grouped[c] for c in present],
            text=present, mode="markers+text", textposition="top center",
            name=DRIFT_LABEL.get(drift, "all"),
            marker=dict(size=13, color=[CELL_COLOUR[c] for c in present]),
            line=dict(dash=DRIFT_DASH.get(drift, "solid"))))
    fig.update_layout(height=400, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      xaxis_title="credits to finish, from the paused position",
                      yaxis_title=f"recall of the true closest {HEADLINE_TOP_FRAC:.0%} of pairs",
                      legend=dict(orientation="h", y=1.12), margin=dict(l=60, r=30, t=40, b=50))
    return fig


# --------------------------------------------------------------------------- RQ2
RQ2_ALPHA = 0.05
RQ2_TAGS = {0.99: "r099", 0.98: "r098", 0.95: "r095", 0.90: "r090"}


def rq2_power_table(run: Run) -> Optional[pd.DataFrame]:
    """Power per (cell, target_rho), recomputed from the saved draws.

    NOT read from ``out/rq2_power.csv``, which is computed with the pipeline's default
    ``cell_fields`` of (num_subjects, screening_min_reliability) and therefore POOLS the two drift
    values into one null and one alternative. That biases power DOWNWARD: the two drift levels have
    different means (at rho=0.90, a ceiling of 0.655 against 0.634), so mixing them widens the null
    and drags its 5th percentile lower, and fewer alternative draws fall below it. At rho=0.90 the
    pooled table reads 0.83-0.90 where the conditioned one reads 0.93-1.00.

    So the critical value is taken WITHIN each drift level, and the two resulting power estimates
    are then averaged. Drift is a simulation parameter, not a finding, and it does not appear in the
    output - but conditioning on it before averaging is what makes the number unbiased. Averaging
    also restores the full 30 draws per cell that the pooled version nominally had.
    """
    null = run.table("rq2_null_draws")
    if null is None:
        return None
    # The alternative arm ran only at the calibrated setting, so the null must be restricted to it
    # or its spread would carry sensitivity axes the alternative never varied.
    cal = run.json("calibration", "calibration") or {}
    dispersion = cal.get("dispersion")
    null = null[null["ndim"] == HEADLINE_NDIM]
    if dispersion is not None:
        null = null[np.isclose(null["perspective_dispersion"], float(dispersion))]
    null = null[np.isclose(null["canvas_softness"], 4.0)]

    keys = ["num_subjects", "screening_min_reliability", "within_session_drift"]
    rows = []
    for rho, tag in RQ2_TAGS.items():
        alt = run.table(f"rq2_alt_draws_{tag}")
        if alt is None:
            continue
        for key, group in alt.groupby(keys):
            mask = np.ones(len(null), dtype=bool)
            for name, value in zip(keys, key):
                mask &= np.isclose(null[name], value)
            cell_null = null[mask]
            if cell_null.empty or group.empty:
                continue
            critical = float(np.quantile(cell_null["spearman"], RQ2_ALPHA))
            rows.append({
                "num_subjects": int(key[0]), "screening_min_reliability": float(key[1]),
                "within_session_drift": float(key[2]), "target_rho": rho,
                "ceiling": float(cell_null["spearman"].mean()),
                "observed": float(group["spearman"].mean()),
                "critical_value": critical, "n_alt": len(group), "n_null": len(cell_null),
                "power": float((group["spearman"] < critical).mean()),
            })
    if not rows:
        return None
    per_drift = pd.DataFrame(rows)
    # Average over drift: it is a nuisance parameter here, and both levels passed the gate.
    return (per_drift
            .groupby(["num_subjects", "screening_min_reliability", "target_rho"], as_index=False)
            .agg(power=("power", "mean"), ceiling=("ceiling", "mean"),
                 observed=("observed", "mean"), n_alt=("n_alt", "sum"),
                 n_null=("n_null", "sum")))


def rq2_power(run: Run) -> go.Figure:
    """Power against the true pre/post similarity, per cell.

    Power is the share of alternative draws below the null's 5th percentile, where the null is two
    cohorts drawn from the SAME ground truth. No normal approximation and no attenuation formula:
    both distributions are simulated.
    """
    frame = rq2_power_table(run)
    if frame is None:
        return _pending("out/rq2_null_draws.csv or rq2_alt_draws_*.csv not found")
    frame = run.cells(frame)
    if frame.empty:
        return _pending("the RQ2 draws hold none of the three compared cells")
    fig = go.Figure()
    for cell in CELL_ORDER:
        rows = frame[frame["cell"] == cell].sort_values("target_rho", ascending=False)
        if rows.empty:
            continue
        # Binomial SE on a proportion of n_alt draws. Shown because 30 draws is coarse and the
        # bars are wide enough to change how a reader weighs small differences between cells.
        se = np.sqrt(rows["power"] * (1 - rows["power"]) / rows["n_alt"].clip(lower=1))
        fig.add_trace(go.Scatter(
            x=rows["target_rho"], y=rows["power"], mode="lines+markers", name=cell,
            error_y=dict(type="data", array=se, visible=True),
            line=dict(color=CELL_COLOUR[cell])))
    fig.add_hline(y=0.80, line_dash="dash", line_color="#d62728",
                  annotation_text="80% power", annotation_position="bottom right")
    fig.update_layout(height=420, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      xaxis_title="true ρ(pre, post) - smaller is a larger SHINE effect",
                      yaxis_title="power", yaxis_range=[0, 1],
                      xaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=60, r=30, t=30, b=90))
    return fig


FIGURES: Dict[str, Fig] = {
    "validation_gate": Fig(
        validation_gate,
        "The six quantities production already shows, against what the recalibrated model "
        "produces. This is the run's licence to be believed about the cells that cannot be "
        "measured: a model that misses what we can check is not worth buying an instance for."),
    "coverage_by_cell": Fig(
        coverage_by_cell,
        "Fraction of the 262,450 image pairs any subject judged. This is the one quantity where N "
        "and screening act in opposite directions - screening discards sessions and so discards "
        "coverage. " + SD_NOTE),
    "recovery_by_cell": Fig(
        recovery_by_cell,
        "Agreement between the recovered embedding and the ground truth it was generated from. "
        + SD_NOTE + " " + DRIFT_NOTE),
    "stability_by_cell": Fig(
        stability_by_cell,
        "Agreement between two independent cohorts in the same cell: what the study would find if "
        "it were run twice. It needs no ground truth, so it is the more trustworthy of the two "
        "recovery measures. " + SD_NOTE),
    "jaccard_by_cell": Fig(
        jaccard_by_cell,
        "Overlap of the closest-pair sets two cohorts recover - the local structure the hierarchy "
        "comparison actually reads, rather than the whole geometry. " + SD_NOTE),
    "cost_per_quality": Fig(
        cost_per_quality,
        "The decision in one panel: what each option costs to finish from the paused position, "
        "against what it buys. Credits are marginal, not total, because only the remainder can "
        "still be spent differently."),
    "rq2_power": Fig(
        rq2_power,
        "Power to detect that ρ(pre, post) sits below the within-condition ceiling, against the "
        "true effect, at the calibrated setting only. Error bars are the binomial SE on 30 draws, "
        "and they are wide: differences between the three cells are mostly within them. The "
        "perturbation is ISOTROPIC by construction, so if SHINE acts selectively on sensory "
        "dimensions - which is what RQ3 supposes - this is optimistic or pessimistic depending on "
        "how much of the distance variance those dimensions carry."),
}
