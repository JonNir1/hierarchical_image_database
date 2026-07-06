"""
Open per-subject trial arrangement grids in the browser.

Run from repo root:
    # all subjects
    .venv/Scripts/python analysis/pilot/show_trials.py

    # filter by task version
    .venv/Scripts/python analysis/pilot/show_trials.py --version 2.0
    .venv/Scripts/python analysis/pilot/show_trials.py --version 1.0

    # single subject -- pass any substring of the session filename
    .venv/Scripts/python analysis/pilot/show_trials.py 15h45
    .venv/Scripts/python analysis/pilot/show_trials.py 2026-06-05

    # combine version filter with filename substring
    .venv/Scripts/python analysis/pilot/show_trials.py --version 2.0 16h22

    # list available session filenames (with versions) without opening anything
    .venv/Scripts/python analysis/pilot/show_trials.py --list
    .venv/Scripts/python analysis/pilot/show_trials.py --version 2.0 --list
"""
import sys
import warnings
from pathlib import Path

# Ensure repo root is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.io as pio

from analysis.pilot.parser import load_pilot_data
from analysis.utils.visualize_trials import visualize_trials

pio.renderers.default = "browser"

with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter("always")
    data = load_pilot_data("data/pilot")

df = data["trials"]

# ---------------------------------------------------------------------------
# Parse arguments (no argparse to keep the script dependency-free)
# ---------------------------------------------------------------------------

args = sys.argv[1:]

version_filter: float | None = None
list_only = False
filename_query: str | None = None

i = 0
while i < len(args):
    arg = args[i]
    if arg == "--version":
        if i + 1 >= len(args):
            print("Error: --version requires a value (e.g. --version 2.0)")
            sys.exit(1)
        try:
            version_filter = float(args[i + 1])
        except ValueError:
            print(f"Error: invalid version '{args[i + 1]}' — must be a number (e.g. 2.0)")
            sys.exit(1)
        i += 2
    elif arg == "--list":
        list_only = True
        i += 1
    elif arg.startswith("--"):
        print(f"Unknown option '{arg}'. Valid options: --version, --list")
        sys.exit(1)
    else:
        filename_query = arg
        i += 1

# ---------------------------------------------------------------------------
# Build the candidate session file list
# ---------------------------------------------------------------------------

# Per-session version lookup (task_version is constant within a session)
session_version = (
    df.groupby("session_file")["task_version"].first().to_dict()
)
all_files = sorted(df["session_file"].unique())

# Apply --version filter
if version_filter is not None:
    available_versions = sorted({v for v in session_version.values()})
    filtered = [f for f in all_files if session_version.get(f) == version_filter]
    if not filtered:
        print(
            f"No subjects found for task version {version_filter:g}.\n"
            f"Available versions: {', '.join(f'v{v:g}' for v in available_versions)}"
        )
        sys.exit(1)
    all_files = filtered

# Apply filename substring filter
if filename_query is not None:
    matches = [f for f in all_files if filename_query in f]
    if not matches:
        scope = f"version {version_filter:g}" if version_filter is not None else "all versions"
        print(f"No session file ({scope}) contains '{filename_query}'.")
        print("Run with --list to see available filenames.")
        sys.exit(1)
    all_files = matches

# ---------------------------------------------------------------------------
# --list mode
# ---------------------------------------------------------------------------

if list_only:
    print("Available session files:")
    for f in all_files:
        v = session_version.get(f)
        version_tag = f"  [v{v:g}]" if v is not None else ""
        print(f"  {f}{version_tag}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Open grids
# ---------------------------------------------------------------------------

for session_file in all_files:
    v = session_version.get(session_file)
    vtag = f" (v{v:g})" if v is not None else ""
    print(f"Opening grid for {session_file}{vtag} …")
    visualize_trials(df[df["session_file"] == session_file]).show()
