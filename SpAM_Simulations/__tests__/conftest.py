import sys
from pathlib import Path

# Make the repo root importable so `import SpAM_Simulations.*` works regardless
# of the working directory pytest is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
