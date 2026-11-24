"""
Random-placement null for SpAM pairwise distances, and a per-trial distance-from-null measure.

Distances are normalised by the unit-square diagonal (sqrt(2)) so the null is directly comparable
to SpAM ``pairwise_distances`` (arena-diagonal normalised).

Usage (from repo root):
    from analysis.utils.null_distances import simulate, trial_ks_distance

    null = simulate(num_dots=20, num_trials=1000, seed=42)
    d = trial_ks_distance(pw_json, null)

    python -m analysis.utils.null_distances --num_dots 20 --num_trials 1000 --seed 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.stats import ks_2samp

from analysis.utils.parser import parse_pairwise_distances


def simulate(num_dots: int, num_trials: int, seed: int = 42) -> np.ndarray:
    """All normalised pairwise distances over *num_trials* simulated trials of *num_dots*
    points placed uniformly at random in the unit square.

    Returns a 1-D array of shape ``(num_trials * C(num_dots, 2),)``.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(num_trials, num_dots, 2))
    diff = pts[:, :, np.newaxis, :] - pts[:, np.newaxis, :, :]  # (T, K, K, 2)
    dist = np.sqrt((diff ** 2).sum(axis=-1))                     # (T, K, K)
    r, c = np.triu_indices(num_dots, k=1)                        # C(K, 2) pairs
    return (dist[:, r, c] / np.sqrt(2)).ravel()


def trial_ks_distance(pw_json: str, null_distribution: np.ndarray) -> float:
    """KS D statistic of one trial's pairwise distances against *null_distribution*.

    NaN when the trial has fewer than two distances.
    """
    dists = list(parse_pairwise_distances(pw_json).values())
    if len(dists) < 2:
        return np.nan
    D, _ = ks_2samp(dists, null_distribution)
    return D


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the SpAM pairwise-distance null distribution.")
    parser.add_argument("--num_dots", type=int, required=True, help="Points placed per trial (e.g. 20)")
    parser.add_argument("--num_trials", type=int, required=True, help="Number of simulated trials (e.g. 1000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    n_pairs = args.num_dots * (args.num_dots - 1) // 2
    print(f"num_dots={args.num_dots}  num_trials={args.num_trials}  -> {args.num_trials * n_pairs:,} distances")

    t0 = time.perf_counter()
    dists = simulate(args.num_dots, args.num_trials, seed=args.seed)
    elapsed = time.perf_counter() - t0

    print(f"Elapsed : {elapsed * 1000:.1f} ms")
    print(f"Mean    : {dists.mean():.4f}")
    print(f"SD      : {dists.std():.4f}")
    print(f"Min/Max : {dists.min():.4f} / {dists.max():.4f}")


if __name__ == "__main__":
    main()
