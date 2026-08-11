"""Build the task-v5 stage-2 report: one self-contained HTML page from a downloaded run.

Every number is read from the run's CSVs at build time rather than transcribed, so the page cannot
drift from the tables it describes. Figures are plotly, inlined, so the file opens anywhere with no
environment and no network.

Usage (from the repo root)::

    python -m SpAM_Simulations.reporting.build_report \\
        --run SpAM_Simulations/sim_results/design-comparison-v5 \\
        --out SpAM_Simulations/sim_results/design-comparison-v5/report_v5.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

ARMS = {0.0: "random", 1.0: "designed"}
ARM_COLOUR = {"random": "#888888", "designed": "#1f77b4",
              "designed_unconstrained": "#9467bd", "pilot": "#d62728", "sim": "#1f77b4"}
LINKAGE_COLOUR = {"average": "#1f77b4", "ward": "#2ca02c", "complete": "#ff7f0e"}
PLOT_BG = "rgba(0,0,0,0)"


# --------------------------------------------------------------------------- loading
class Run:
    """Every table a report section might want, with the missing ones simply absent."""

    NAMES = ("design_only", "coverage", "stability", "cluster_agreement", "dendrogram_agreement",
             "cluster_sizes", "k_selection", "density_agreement", "embedding_stability",
             "embedding_generalizability", "topk_jaccard", "recovery_vs_gt",
             "validity_gradient", "noise_vs_distance", "noise_curve_shape")
    DIAGNOSTICS = ("level_coverage", "gt_gradient", "gt_vs_raw", "noise_ceiling")

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.tables: Dict[str, pd.DataFrame] = {}
        for name in self.NAMES:
            path = self.dir / "out" / f"{name}.csv"
            if path.is_file():
                frame = pd.read_csv(path)
                if "allocation_mode" in frame.columns and "arm" not in frame.columns:
                    frame["arm"] = frame["allocation_mode"].map(ARMS)
                self.tables[name] = frame
        for name in self.DIAGNOSTICS:
            path = self.dir / "gt_diagnostics" / f"{name}.csv"
            if path.is_file():
                self.tables[name] = pd.read_csv(path)
        cal = self.dir / "calibration" / "calibration.json"
        self.calibration = json.loads(cal.read_text()) if cal.is_file() else {}

    def get(self, name: str) -> Optional[pd.DataFrame]:
        return self.tables.get(name)

    def has(self, *names: str) -> bool:
        return all(n in self.tables for n in names)


# --------------------------------------------------------------------------- page furniture
def _fig(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=height,
        margin=dict(l=60, r=30, t=55, b=55), paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(size=12), legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="closest")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def _by_arm_lines(frame: pd.DataFrame, value: str, title: str, ylab: str,
                  x: str = "num_subjects", arm_col: str = "arm", log_x: bool = True,
                  hline: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    for arm, sub in frame.groupby(arm_col, sort=False):
        agg = sub.groupby(x)[value].agg(["mean", "std", "count"]).reset_index()
        sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        fig.add_trace(go.Scatter(
            x=agg[x], y=agg["mean"], mode="lines+markers", name=str(arm),
            line=dict(color=ARM_COLOUR.get(str(arm)), width=2), marker=dict(size=8),
            error_y=dict(type="data", array=sem, visible=True, thickness=1)))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(dash="dot", color="rgba(128,128,128,0.7)"))
    fig.update_xaxes(title_text="participants (N)", type="log" if log_x else "linear",
                     tickvals=sorted(frame[x].unique()))
    fig.update_yaxes(title_text=ylab)
    return _fig(fig, title)


def _table_html(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    out = frame.copy()
    for col in out.select_dtypes(include=[float]).columns:
        out[col] = out[col].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return out.to_html(index=False, border=0, classes="datatable", escape=False)


def _section(anchor: str, title: str, body: str, figures: Sequence[go.Figure] = ()) -> str:
    parts = [f'<section id="{anchor}"><h2>{html.escape(title)}</h2>', body]
    for fig in figures:
        parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False,
                                 config={"displayModeBar": False}))
    parts.append("</section>")
    return "\n".join(parts)


def _missing(name: str) -> str:
    return (f'<p class="warn">Table <code>{html.escape(name)}</code> was not present in this run, '
            f'so this section could not be built.</p>')


def _pct(x: float) -> str:
    return f"{x:.1f}%"


CSS = """
:root { color-scheme: light dark; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       line-height: 1.65; color: #1a1a1a; background: #ffffff; }
.wrap { max-width: 960px; margin: 0 auto; padding: 2rem 1.25rem 5rem; }
h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 .4rem; }
h2 { font-size: 1.35rem; margin: 3rem 0 .75rem; padding-top: 1rem;
     border-top: 1px solid rgba(128,128,128,.28); }
h3 { font-size: 1.05rem; margin: 1.75rem 0 .5rem; }
.sub { color: #666; margin: 0 0 2rem; font-size: .95rem; }
p { margin: .8rem 0; }
ul { margin: .8rem 0; padding-left: 1.3rem; }
li { margin: .4rem 0; }
code { background: rgba(128,128,128,.14); padding: .1em .35em; border-radius: 3px;
       font-size: .9em; }
.note { border-left: 3px solid #1f77b4; background: rgba(31,119,180,.07);
        padding: .7rem 1rem; margin: 1.2rem 0; border-radius: 0 4px 4px 0; }
.warn { border-left: 3px solid #d62728; background: rgba(214,39,40,.07);
        padding: .7rem 1rem; margin: 1.2rem 0; border-radius: 0 4px 4px 0; }
table { border-collapse: collapse; margin: 1.2rem 0; font-size: .88rem; width: 100%;
        display: block; overflow-x: auto; }
th, td { padding: .4rem .7rem; text-align: right; border-bottom: 1px solid rgba(128,128,128,.2); }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid rgba(128,128,128,.4); font-weight: 600; }
table.kv th { text-align: left; width: 30%; font-weight: 600; }
nav { background: rgba(128,128,128,.08); padding: 1rem 1.25rem; border-radius: 6px;
      margin: 1.5rem 0 2rem; }
nav ol { margin: .3rem 0; padding-left: 1.4rem; }
nav a { color: #1f77b4; text-decoration: none; }
nav a:hover { text-decoration: underline; }
@media (prefers-color-scheme: dark) {
  body { color: #e8e8e8; background: #14161a; }
  .sub { color: #9aa0a6; }
  nav a { color: #6db3f2; }
}
"""

TOC = [
    ("model", "What was simulated"),
    ("arms", "The arms, and what is merely swept"),
    ("gt", "The ground truth, and the limit it puts on everything"),
    ("coverage", "Does the design see more image pairs?"),
    ("deploy", "What the deployable constraint costs"),
    ("reliability", "Reliability before any embedding"),
    ("embedding", "Does the embedding itself reproduce?"),
    ("recovery", "Recovering the ground truth"),
    ("clusters", "At what granularity does structure reproduce?"),
    ("density", "Which images belong to no group at all?"),
    ("validity", "Is the simulation realistic enough to believe?"),
    ("limits", "What this can and cannot answer"),
]


def build(run: Run, title: str = "SpAM design simulation (task-v5, stage 2)") -> str:
    """Assemble the whole page."""
    from SpAM_Simulations.reporting import report_clusters as rc
    from SpAM_Simulations.reporting import report_sections as rs

    sections = [
        rs.section_model(run), rs.section_arms(run), rs.section_ground_truth(run),
        rs.section_coverage(run), rs.section_deployability(run), rs.section_reliability(run),
        rs.section_embedding(run), rs.section_recovery(run),
        rc.section_clusters(run), rc.section_density(run), rc.section_validity(run),
        rc.section_limits(run),
    ]
    toc = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in TOC)
    missing = [n for n in ("embedding_stability", "recovery_vs_gt") if not run.has(n)]
    banner = ("" if not missing else
              f'<p class="warn">Built without {", ".join(missing)} &mdash; those sections are '
              f'placeholders.</p>')
    # The whole library, inlined. A CDN reference would make the page useless offline, and
    # slicing it out of a rendered figure (an earlier attempt) silently produced a 0.2 MB page
    # with 18 figures and no plotting library to draw them.
    plotly_js = f"<script>{get_plotlyjs()}</script>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
{plotly_js}
</head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">Simulation report &mdash; does a designed image-to-trial allocation recover the
perceptual space better than the random allocation currently deployed?</p>
{banner}
<nav><strong>Contents</strong><ol>{toc}</ol></nav>
{"".join(sections)}
</div></body></html>"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="downloaded run directory")
    p.add_argument("--out", type=Path, default=None, help="defaults to <run>/report_v5.html")
    p.add_argument("--title", default="SpAM design simulation (task-v5, stage 2)")
    args = p.parse_args(argv)

    run = Run(args.run)
    print(f"[report] tables loaded: {sorted(run.tables)}")
    absent = [n for n in Run.NAMES if n not in run.tables]
    if absent:
        print(f"[report] NOT present: {absent}")
    out = args.out or (args.run / "report_v5.html")
    out.write_text(build(run, args.title), encoding="utf-8")
    print(f"[report] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
