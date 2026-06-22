"""
Simulate the null distribution of pairwise Euclidean distances when num_dots
points are placed uniformly at random in the 2D unit square, repeated
num_trials times.

Distances are normalised by the unit-square diagonal (sqrt(2)) so the output
is directly comparable to SpAM pairwise_distances (arena-diagonal normalised).

Usage:
    python analysis/pilot/simulate_null_distances.py --num_dots 20 --num_trials 1000
    python analysis/pilot/simulate_null_distances.py --num_dots 20 --num_trials 500 --seed 0
"""
import argparse
import time

import numpy as np


def simulate(num_dots: int, num_trials: int, seed: int = 42) -> np.ndarray:
    """
    Returns a 1-D array of shape (num_trials × C(num_dots, 2),) containing all
    normalised pairwise distances across num_trials simulated trials.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(num_trials, num_dots, 2))
    diff = pts[:, :, np.newaxis, :] - pts[:, np.newaxis, :, :]  # (T, K, K, 2)
    dist = np.sqrt((diff ** 2).sum(axis=-1))                     # (T, K, K)
    r, c = np.triu_indices(num_dots, k=1)                        # C(K,2) pairs
    return (dist[:, r, c] / np.sqrt(2)).ravel()                  # normalise by diagonal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate pairwise distance null distribution for SpAM."
    )
    parser.add_argument("--num_dots", type=int, required=True,
                        help="Points placed per trial (e.g. 20)")
    parser.add_argument("--num_trials", type=int, required=True,
                        help="Number of simulated trials (e.g. 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    args = parser.parse_args()

    n_pairs = args.num_dots * (args.num_dots - 1) // 2
    print(f"num_dots={args.num_dots}  num_trials={args.num_trials}  -> {args.num_trials * n_pairs:,} distances")

    t0 = time.perf_counter()
    dists = simulate(args.num_dots, args.num_trials, seed=args.seed)
    elapsed = time.perf_counter() - t0

    print(f"Elapsed : {elapsed*1000:.1f} ms")
    print(f"Mean    : {dists.mean():.4f}")
    print(f"SD      : {dists.std():.4f}")
    print(f"Min/Max : {dists.min():.4f} / {dists.max():.4f}")


if __name__ == "__main__":
    main()
