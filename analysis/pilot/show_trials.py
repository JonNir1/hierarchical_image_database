"""
Open per-subject trial arrangement grids in the browser.

Run from repo root:
    # all subjects
    .venv/Scripts/python analysis/pilot/show_trials.py

    # filter by task version
    .venv/Scripts/python analysis/pilot/show_trials.py --version 2.0
    .venv/Scripts/python analysis/pilot/show_trials.py --version 1.0

    # single subject -- by participant_id (exact or substring match)
    .venv/Scripts/python analysis/pilot/show_trials.py --participant 6a0f812b04e94d1ec3d4bd0c
    .venv/Scripts/python analysis/pilot/show_trials.py -p 6a0f812b

    # single subject -- pass any substring of the session filename
    .venv/Scripts/python analysis/pilot/show_trials.py 15h45
    .venv/Scripts/python analysis/pilot/show_trials.py 2026-06-05

    # combine filters (version + participant + filename substring all AND together)
    .venv/Scripts/python analysis/pilot/show_trials.py --version 2.0 16h22
    .venv/Scripts/python analysis/pilot/show_trials.py --version 3.0 -p 6a0f812b

    # list available session filenames (with versions and participant_ids) without
    # opening anything
    .venv/Scripts/python analysis/pilot/show_trials.py --list
    .venv/Scripts/python analysis/pilot/show_trials.py --version 2.0 --list
"""
import sys
import warnings
from pathlib import Path

# Ensure repo root is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.io as pio

from analysis.utils.parser_v2 import load_data
from analysis.utils.visualize_trials import visualize_trials

pio.renderers.default = "browser"

with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter("always")
    data = load_data("data")

# parser_v2 splits the old single frame in two: per-trial rows, and per-participant
# session metadata. Re-join the two session-level fields this script filters on, keeping
# the old `session_file` name so the rest of the script is unchanged.
df = data["trials"].merge(
    data["participants"][["participant_id", "file_name", "task_version", "cohort"]],
    on="participant_id", how="left",
).rename(columns={"file_name": "session_file"})

# ---------------------------------------------------------------------------
# Parse arguments (no argparse to keep the script dependency-free)
# ---------------------------------------------------------------------------

args = sys.argv[1:]

version_filter: float | None = None
participant_filter: str | None = None
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
    elif arg in ("--participant", "-p"):
        if i + 1 >= len(args):
            print(f"Error: {arg} requires a value (e.g. {arg} 6a0f812b)")
            sys.exit(1)
        participant_filter = args[i + 1]
        i += 2
    elif arg == "--list":
        list_only = True
        i += 1
    elif arg.startswith("-"):
        print(f"Unknown option '{arg}'. Valid options: --version, --participant/-p, --list")
        sys.exit(1)
    else:
        filename_query = arg
        i += 1

# ---------------------------------------------------------------------------
# Build the candidate session file list
# ---------------------------------------------------------------------------

# Per-session version / participant_id lookup (both constant within a session)
session_version = (
    df.groupby("session_file")["task_version"].first().to_dict()
)
session_participant = (
    df.groupby("session_file")["participant_id"].first().to_dict()
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

# Apply --participant filter (substring match against participant_id)
if participant_filter is not None:
    matches = [f for f in all_files if participant_filter in session_participant.get(f, "")]
    if not matches:
        print(f"No session found with participant_id containing '{participant_filter}'.")
        print("Run with --list to see available participant_ids.")
        sys.exit(1)
    all_files = matches

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
        pid = session_participant.get(f)
        pid_tag = f"  participant={pid}" if pid is not None else ""
        print(f"  {f}{version_tag}{pid_tag}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Open grids
# ---------------------------------------------------------------------------

for session_file in all_files:
    v = session_version.get(session_file)
    vtag = f" (v{v:g})" if v is not None else ""
    print(f"Opening grid for {session_file}{vtag} …")
    visualize_trials(df[df["session_file"] == session_file]).show()
