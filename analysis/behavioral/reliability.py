"""
Noise-ceiling reliability comparison for the three behavioral-RDM aggregation
methods ('mean' / 'simple' / 'evidence', see analysis/rdms/behavioral.py),
following the lower/upper-bound noise-ceiling logic of Nili et al. (2014):

  lower bound -- leave-one-participant-out (LOO): exact, N folds (one per
      retained participant, no sampling). The left-out side is always the
      plain per-pair mean of that one participant's own trials (see
      loo_reliability()'s docstring for why: the iterative evidence/simple
      rescale has essentially no cross-trial calibration signal within a
      single person's own ~18 non-overlapping trial groups). The "others"
      side is combine_method(all other participants), for whichever
      aggregation is under test.
  upper bound -- split-half: S random disjoint participant-set halves,
      each combined via the method under test, compared to each other.
      The same S splits are reused across all three aggregation methods
      (see split_half_reliability()'s `splits` argument), so the comparison
      is paired/matched at each split index rather than confounded by which
      participants happened to land in which half for a given method.

Both bounds are scored with Kendall's tau (the leading metric) and Spearman's
rho (secondary/consistency check), computed directly on the condensed RDM
vectors over pair positions non-NaN in both sides -- no MDS/embedding step is
involved anywhere in this module. The split-half upper bound's raw Spearman is
also Spearman-Brown-projected to full-sample size (`ceiling_full` column,
mirroring SpAM_Simulations/empirical/gt_diagnostics.py's raw_noise_ceiling())
for direct comparability with that module's own prior real-data noise-ceiling
numbers (SpAM_Simulations/sim_results/v5/prod/prod_noise_ceiling.csv); it is
NaN everywhere else -- not computed for Kendall (the correction isn't
well-justified for a rank-concordance statistic) or for the lower bound (LOO
already uses N-1 participants on the "others" side, not a half, so there's no
half-to-full projection to make). Procrustes/MDS-based comparison was
considered and dropped: every retained participant's own trial-derived pair
graph is disconnected into ~18 separate components (confirmed by direct
computation against the real cached data -- each of a participant's 22
non-catch trials draws from a non-overlapping slice of their fixed 360-image
pool), so a single left-out participant's own data can never be embedded via
MDS at all, independent of aggregation method or NaN-handling.

The pairwise significance testing (evidence-vs-mean primary; evidence-vs-simple,
simple-vs-mean secondary) and the summary figure are deliberately NOT in this
module -- they're notebook work, see analysis/behavioral/noise_ceiling.ipynb,
matching this repo's existing notebook-as-analysis-surface convention
(SpAM_Simulations/evaluation_v0_1.ipynb, visualize_dataset/dataset_visualizations.ipynb).
This module only produces the underlying tidy long-format reliability data.

Usage (from repo root, with the main checkout's .venv active):
    python -m analysis.behavioral.reliability --cache-dir <path to analysis/results/parsed_data>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rdms.behavioral import (
    _AGGREGATIONS, _combine, _mean_combine, _select_participants, _trial_sparse,
)
from analysis.rdms.common import REPO_ROOT, image_manifest_index
from analysis.utils.load_data import load_data
from analysis.utils.stats import masked_rank_correlation

_VALID_VARIANTS = ("pre_shine", "post_shine")
_DEFAULT_CACHE_DIR = REPO_ROOT / "analysis" / "results" / "parsed_data"
RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "behavioral"
_DEFAULT_MIN_OVERLAP = 20
_DEFAULT_N_SPLITS = 100


def _cohort_name(variant: str) -> str:
    return "pre" if variant == "pre_shine" else "post"


def _spearman_brown(r: float) -> float:
    """Spearman-Brown projection of a half-sample correlation to full-sample size:
    2r / (1 + r). Mirrors SpAM_Simulations/empirical/gt_diagnostics.py's raw_noise_ceiling()
    (gt_diagnostics.py:~168), including its guard: a non-positive half-split correlation makes
    the projection meaningless, so it's passed through unchanged rather than "corrected" into a
    confident-looking number."""
    if not np.isfinite(r) or r <= 0:
        return r
    return float(2 * r / (1 + r))


def _add_ceiling_full(long: pd.DataFrame) -> pd.DataFrame:
    """Add the Spearman-Brown-projected `ceiling_full` column to a melted long-format
    reliability frame (columns include `bound`, `metric`, `value`). NaN outside
    bound='upper', metric='spearman' -- see _spearman_brown()'s docstring and the module
    docstring for why the projection is restricted to that combination."""
    is_upper_spearman = (long["bound"] == "upper") & (long["metric"] == "spearman")
    long = long.copy()
    long["ceiling_full"] = np.where(
        is_upper_spearman, long["value"].map(_spearman_brown), np.nan,
    )
    return long


def _parse_trials_by_participant(
    df_trials: pd.DataFrame,
    participant_ids: pd.Series | list,
    variant: str,
    image_to_idx: dict[str, int],
    n_conds: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Parse every non-catch trial for the given participants exactly once, grouped by
    participant_id -- so LOO/split-half folds can select/concatenate sublists from this
    cache instead of re-parsing pairwise_distances JSON once per fold (see the module's
    plan: with N + S ~= 179-200 folds x 3 aggregators, re-parsing per fold would redundantly
    redo the same work hundreds of times)."""
    rows = df_trials[
        df_trials["participant_id"].isin(participant_ids) & ~df_trials["is_catch"]
    ]
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {pid: [] for pid in participant_ids}
    for pid, pw in zip(rows["participant_id"], rows["pairwise_distances"]):
        parsed = _trial_sparse(pw, variant, image_to_idx, n_conds)
        if parsed is not None:
            out[pid].append(parsed)
    return out


def loo_reliability(
    variant: str,
    aggregation: str,
    *,
    trials_by_participant: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    n_conds: int,
    min_overlap: int = _DEFAULT_MIN_OVERLAP,
) -> pd.DataFrame:
    """Leave-one-participant-out lower bound: one row per held-out participant.

    Left-out side is always _mean_combine([p's own trials]) regardless of `aggregation` --
    see the module docstring for why applying the method-under-test to a single participant's
    data wouldn't reflect the effect being studied. Others side is combine_method(everyone
    else), for the aggregation under test.
    """
    participant_ids = list(trials_by_participant.keys())
    rows = []
    for pid in participant_ids:
        left_trials = trials_by_participant[pid]
        others_trials = [t for other in participant_ids if other != pid for t in trials_by_participant[other]]
        left_condensed, _ = _mean_combine(left_trials, n_conds)
        others_condensed, _ = _combine(others_trials, n_conds, aggregation)
        stats = masked_rank_correlation(left_condensed, others_condensed, min_overlap=min_overlap)
        rows.append({"fold_id": pid, **stats})
    return pd.DataFrame(rows)


def make_splits(
    participant_ids: list[str], n_splits: int, seed: int,
) -> list[tuple[list[str], list[str]]]:
    """S disjoint half-partitions of participant_ids, one fixed random split per draw.

    Odd N splits as floor(N/2) / ceil(N/2). Reused identically across all three aggregation
    methods by the caller (pass the same `splits` list to every split_half_reliability() call
    for a given variant), matching SpAM_Simulations/empirical/gt_construction.py's
    dimensionality_scan() pattern of reusing the same splits across every dimensionality.
    """
    rng = np.random.default_rng(seed)
    n = len(participant_ids)
    half = n // 2
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(participant_ids)
        splits.append((list(perm[:half]), list(perm[half:])))
    return splits


def split_half_reliability(
    variant: str,
    aggregation: str,
    *,
    trials_by_participant: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    n_conds: int,
    splits: list[tuple[list[str], list[str]]],
    min_overlap: int = _DEFAULT_MIN_OVERLAP,
) -> pd.DataFrame:
    """Split-half upper bound: one row per split index, using pre-generated `splits`
    (see make_splits()) so the same participant partitions are reused across aggregators."""
    rows = []
    for split_id, (half_a, half_b) in enumerate(splits):
        trials_a = [t for pid in half_a for t in trials_by_participant[pid]]
        trials_b = [t for pid in half_b for t in trials_by_participant[pid]]
        condensed_a, _ = _combine(trials_a, n_conds, aggregation)
        condensed_b, _ = _combine(trials_b, n_conds, aggregation)
        stats = masked_rank_correlation(condensed_a, condensed_b, min_overlap=min_overlap)
        rows.append({"fold_id": split_id, **stats})
    return pd.DataFrame(rows)


def run_comparison(
    variant: str,
    *,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    n_splits: int = _DEFAULT_N_SPLITS,
    seed: int = 0,
    min_overlap: int = _DEFAULT_MIN_OVERLAP,
) -> pd.DataFrame:
    """Run LOO + split-half reliability for all three aggregation methods on one variant.

    Returns a tidy long-format DataFrame with columns
    [cohort, aggregation, bound, fold_id, metric, value, n_overlap, ceiling_full], `metric` in
    {'kendall', 'spearman'}, `bound` in {'lower', 'upper'}. `ceiling_full` is the Spearman-Brown
    projection of `value` to full-sample size; NaN except for bound='upper', metric='spearman'
    rows (see module docstring).
    """
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant must be 'pre_shine' or 'post_shine', got {variant!r}")

    data = load_data(cache_dir, prod_only=True)
    participants = _select_participants(data["participants"], variant)
    participant_ids = list(participants["participant_id"])

    image_to_idx, n_conds = image_manifest_index()

    trials_by_participant = _parse_trials_by_participant(
        data["trials"], participant_ids, variant, image_to_idx, n_conds,
    )
    print(f"[reliability] variant={variant!r}: {len(participant_ids)} participants, "
          f"{sum(len(v) for v in trials_by_participant.values())} non-catch trials parsed once.")

    splits = make_splits(participant_ids, n_splits, seed)
    cohort = _cohort_name(variant)

    frames = []
    for aggregation in _AGGREGATIONS:
        loo = loo_reliability(
            variant, aggregation, trials_by_participant=trials_by_participant,
            n_conds=n_conds, min_overlap=min_overlap,
        )
        loo["bound"] = "lower"
        half = split_half_reliability(
            variant, aggregation, trials_by_participant=trials_by_participant,
            n_conds=n_conds, splits=splits, min_overlap=min_overlap,
        )
        half["bound"] = "upper"
        wide = pd.concat([loo, half], ignore_index=True)
        wide["aggregation"] = aggregation
        wide["cohort"] = cohort
        long = wide.melt(
            id_vars=["cohort", "aggregation", "bound", "fold_id", "n_overlap", "underpowered"],
            value_vars=["kendall", "spearman"],
            var_name="metric", value_name="value",
        )
        frames.append(_add_ceiling_full(long))
        n_underpowered = int(wide["underpowered"].sum())
        if n_underpowered:
            print(f"[reliability] WARNING: {n_underpowered} {aggregation!r} folds had fewer "
                  f"than {min_overlap} jointly-covered pairs and were flagged underpowered.")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--n-splits", type=int, default=_DEFAULT_N_SPLITS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-overlap", type=int, default=_DEFAULT_MIN_OVERLAP)
    parser.add_argument("--only", nargs="+", choices=list(_VALID_VARIANTS), default=list(_VALID_VARIANTS))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for variant in args.only:
        df = run_comparison(
            variant, cache_dir=args.cache_dir, n_splits=args.n_splits,
            seed=args.seed, min_overlap=args.min_overlap,
        )
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    out_path = RESULTS_DIR / "reliability.csv"
    result.to_csv(out_path, index=False)
    print(f"[reliability] Saved {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
