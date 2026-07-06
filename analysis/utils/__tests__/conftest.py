import sys
from pathlib import Path

# Make the repo root importable so 'from analysis.utils.visualize_trials import ...'
# works regardless of the working directory pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
