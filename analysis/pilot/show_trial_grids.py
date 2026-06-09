"""
Open per-subject trial arrangement grids in the browser.

Run from repo root:
    # all subjects
    .venv/Scripts/python analysis/pilot/show_trial_grids.py

    # single subject -- pass any substring of the session filename
    .venv/Scripts/python analysis/pilot/show_trial_grids.py 15h45
    .venv/Scripts/python analysis/pilot/show_trial_grids.py 2026-06-05

    # list available session filenames without opening anything
    .venv/Scripts/python analysis/pilot/show_trial_grids.py --list
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
all_files = sorted(df["session_file"].unique())

if len(sys.argv) > 1 and sys.argv[1] == "--list":
    print("Available session files:")
    for f in all_files:
        print(" ", f)
    sys.exit(0)

if len(sys.argv) > 1:
    query = sys.argv[1]
    matches = [f for f in all_files if query in f]
    if not matches:
        print(f"No session file contains '{query}'.")
        print("Run with --list to see available filenames.")
        sys.exit(1)
    selected = matches
else:
    selected = all_files

for session_file in selected:
    print(f"Opening grid for {session_file} …")
    plot_trial_grid(df[df["session_file"] == session_file]).show()
