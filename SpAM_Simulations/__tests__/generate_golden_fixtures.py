"""Generate the bit-exact golden reference fixtures.

Run this ONCE against the pre-refactor code, then commit the resulting
``fixtures/golden_experiment.npz``. ``test_bit_exact.py`` regenerates the same
entries against the (refactored) code and asserts they are byte-for-byte identical.

Usage (from the repo root, with the project venv):

    python -m SpAM_Simulations.__tests__.generate_golden_fixtures

(or run this file directly; it inserts the repo root onto sys.path).
"""
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script: ensure the repo root (for the SpAM_Simulations
# package) and this test directory (for the _golden_config sibling) are importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _golden_config as gc  # noqa: E402
from SpAM_Simulations.models.experiment import simulate_experiment  # noqa: E402
from SpAM_Simulations.core.simulation import Simulation  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "golden_experiment.npz"


def build_payload() -> dict:
    """Run the simulation for every golden entry and collect the arrays."""
    sim = Simulation.make(gc.N_IMAGES, gc.N_DIMS, seed=gc.GT_SEED)
    gt_distances = sim.gt_distances  # float32 condensed vector

    payload = {
        "gt_distances": gt_distances,
        "numpy_version": np.asarray(np.__version__),
    }
    for combo_idx, rep, params, global_seed, rng_seed in gc.entries():
        np.random.seed(global_seed)
        rng = np.random.default_rng(rng_seed)
        _, res = simulate_experiment(params, gt_distances, rng, verbose=False)
        key = gc.entry_key(combo_idx, rep)
        payload[f"{key}_distances"] = res.distances
        payload[f"{key}_num_obs"] = res.num_obs
        payload[f"{key}_subject_noises"] = res.subject_noises
    return payload


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    np.savez_compressed(FIXTURE_PATH, **payload)
    n_entries = sum(1 for _ in gc.entries())
    print(f"Wrote {FIXTURE_PATH} ({n_entries} entries, numpy {np.__version__})")


if __name__ == "__main__":
    main()
