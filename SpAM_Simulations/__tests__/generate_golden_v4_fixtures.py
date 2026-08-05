"""Generate the task-v4 bit-exact golden reference fixtures.

Run this ONCE against the code as it stands *before* the allocation refactor, then commit the
resulting ``fixtures/golden_task_v4.npz``. ``test_bit_exact_v4.py`` regenerates the same entries
against the refactored code and asserts they are byte-for-byte identical, which is what proves the
default (random) allocation path still consumes the RNG stream in exactly the same order.

Usage (from the repo root, with the project venv):

    python -m SpAM_Simulations.__tests__.generate_golden_v4_fixtures

(or run this file directly; it inserts the repo root onto sys.path).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _golden_v4_config as gc  # noqa: E402
from SpAM_Simulations.simulation import build_ground_truth_embeddings  # noqa: E402
from SpAM_Simulations.task_v4_experiment import simulate_task_v4_experiment  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "golden_task_v4.npz"


def build_payload() -> dict:
    """Run the v4 simulation for every golden entry and collect the arrays."""
    gt = build_ground_truth_embeddings(gc.N_IMAGES, gc.N_DIMS, seed=gc.GT_SEED)
    payload = {
        "gt_embeddings": gt,
        "numpy_version": np.asarray(np.__version__),
    }
    for combo_idx, rep, params, global_seed, rng_seed in gc.entries():
        np.random.seed(global_seed)
        rng = np.random.default_rng(rng_seed)
        _, res = simulate_task_v4_experiment(params, gt, rng, verbose=False)
        key = gc.entry_key(combo_idx, rep)
        payload[f"{key}_distances"] = res.distances
        payload[f"{key}_num_obs"] = res.num_obs
        payload[f"{key}_subject_noises"] = res.subject_noises
        payload[f"{key}_test_retest"] = res.subject_test_retest
        # The retry loop consumes RNG draws, so the recruitment count is itself a stream witness.
        payload[f"{key}_n_candidates"] = np.asarray(res.n_candidates_screened)
    return payload


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    np.savez_compressed(FIXTURE_PATH, **payload)
    n_entries = sum(1 for _ in gc.entries())
    print(f"Wrote {FIXTURE_PATH} ({n_entries} entries, numpy {np.__version__})")


if __name__ == "__main__":
    main()
