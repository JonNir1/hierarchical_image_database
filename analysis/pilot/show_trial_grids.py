"""
Open per-subject trial arrangement grids in the browser.

Run from repo root:
    .venv/Scripts/python analysis/pilot/show_trial_grids.py
"""
import sys
import warnings
from pathlib import Path

# Ensure repo root is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.io as pio

from analysis.pilot.parser import load_pilot_data
from analysis.pilot.visualize import plot_trial_grid

pio.renderers.default = "browser"

with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter("always")
    data = load_pilot_data("data/pilot")

df = data["trials"]

for pid, df_subject in df.groupby("participant_id"):
    print(f"Opening grid for {pid} …")
    plot_trial_grid(df_subject).show()
