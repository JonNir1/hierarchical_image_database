"""
Behavioral RDMs: combine raw SpAM (multi-arrangement) trial data -- partial
pairwise distances covering only the ~20 images shown in each trial -- into one
full 725x725 RDM per SHINE variant, via THREE independent aggregation methods
(pass --aggregation, or build all):

  "evidence" -- iterative per-trial evidence-weighted rescaling before
      averaging (Kriegeskorte & Mur, 2012, https://doi.org/10.3389/fpsyg.2012.00245),
      as implemented in rsatoolbox (van den Bosch et al., 2025,
      https://doi.org/10.1101/2025.05.22.655542):
      rsatoolbox.rdm.combine.from_partials()/rescale()/RDMs.mean(). See
      _sparse_combine() below.
  "simple" -- the same iterative rescaling as 'evidence', but with uniform
      (unweighted) per-trial weighting instead of squared-distance weighting --
      still NOT the same as 'mean' below, since the rescaling step itself still
      runs. Also from rsatoolbox.rdm.combine.rescale(); see _sparse_combine().
      (_sparse_combine() additionally accepts rsatoolbox's third rescale()
      weighting, 'setsize', for internal/test use -- with this project's
      constant trial size it's numerically identical to 'simple', so it isn't
      separately exposed via build_behavioral_rdm()'s public `aggregation`.)
  "mean" -- plain unweighted average of every trial observation touching a pair,
      no rescaling at all: the scheme used by the PiCS dataset (Robbins et al.,
      2025, https://doi.org/10.3758/s13428-025-02732-0) and by this project's own
      SpAM_Simulations pipeline (mean_from_sum_and_count() in
      SpAM_Simulations/core/helpers.py, via _calculate_mean_distances() in
      SpAM_Simulations/measures/metrics.py). See _mean_combine() below.

'evidence' and 'mean' are built (not just rsatoolbox's) because a real bug was found in
rsatoolbox.util.weighted_mds (the MDS step, not this combination step -- see
SpAM_Simulations/core/mds_rsatoolbox_compare.py) that specifically affects data
with missing/zero-weight pairs, which is the norm here. Rather than trust
rsatoolbox's combination algorithm blindly, both methods' RDMs are built so the
downstream MDS embeddings can be compared directly, and so results don't rest
on one library's correctness at a scale nobody else has exercised it at.

_sparse_combine() ("evidence"/"setsize"/"simple") reimplements
rsatoolbox.rdm.combine.from_partials()/rescale() rather than calling them
directly: those materialize a dense (n_trials, 262_450) float64 array
regardless of how sparse each trial is (~190 of 262_450 pairs), which is fine
for the toolbox's typical small-multi-arrangement use case but does not scale
here -- at this project's real data size (~1,700+ trials) that array alone is
>3.5GB, with several more same-shape temporaries live during rescale()'s
iterative loop, enough to push a single process past 10GB RSS on a 16GB
machine without finishing. _sparse_combine() runs the exact same algorithm
(verified against rsatoolbox's own from_partials()+rescale()+mean() on
synthetic data in __tests__/test_behavioral.py) over each trial's own
(indices, values) pairs plus one dense (262_450,) running estimate, instead of
a dense (n_trials, 262_450) array -- no behavior change, only memory footprint
(rsatoolbox itself is still a required dependency, used for that equivalence
check).

Outputs (to analysis/results/rdms/), one pair per aggregation method:
    D_behav_pre_<aggregation>.npy   -- pre-SHINE
    D_behav_post_<aggregation>.npy  -- post-SHINE

Only production-cohort participants with status == "full data" (i.e. passed
screening, or screening was disabled) and task_version >= 4.0 (the v3.x pilot
task had a different block/screening structure) are included. Both the
screening and experimental blocks of these participants' sessions are pooled
-- see the project memory note on this: a subject who passed screening
contributes real perceptual judgements from both blocks, not just the
experimental one.

Usage (from repo root, with the main checkout's .venv active):
    python -m analysis.rdms.behavioral --cache-dir <path to analysis/results/parsed_data>
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from analysis.rdms.common import REPO_ROOT, load_manifest, save_rdm
from analysis.utils.load_data import load_data
from analysis.utils.parser import parse_pairwise_distances

_VALID_VARIANTS = ("pre_shine", "post_shine")
_MIN_TASK_VERSION = 4.0

# Conventional cache location (see analysis/utils/load_data.py). Not present in a
# worktree checkout -- data/ and its parsed-parquet cache live only at the main
# checkout root and are not copied into worktrees; pass --cache-dir to override.
_DEFAULT_CACHE_DIR = REPO_ROOT / "analysis" / "results" / "parsed_data"


def _curated_path_from_src(src: str, variant: str) -> str:
    """Strip the '.../<variant>/' stimuli_path prefix, leaving the manifest curated_path suffix.

    src is the image path string jsPsych recorded in pairwise_distances, e.g.
    "./images/pre_shine/inanimate/handmade/kitchen/bottle1.png" -> "inanimate/handmade/kitchen/bottle1.png".
    """
    marker = f"/{variant}/"
    idx = src.find(marker)
    if idx == -1:
        raise ValueError(f"Could not find {marker!r} in this trial's pairwise_distances src field: {src!r}")
    return src[idx + len(marker):]


def _select_participants(df_participants: pd.DataFrame, variant: str) -> pd.DataFrame:
    if not variant.endswith("_shine"):
        raise ValueError(f"variant must end with '_shine' (e.g. 'pre_shine'), got {variant!r}")
    shine_value = variant.removesuffix("_shine")  # "pre_shine" -> "pre", "post_shine" -> "post"
    mask = (
        (df_participants["status"] == "full data")
        & (df_participants["shine_variant"] == shine_value)
        & (pd.to_numeric(df_participants["task_version"], errors="coerce") >= _MIN_TASK_VERSION)
    )
    return df_participants[mask]


def _condensed_index(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """scipy squareform's condensed-vector index for pairs (i, j), i < j required."""
    return n * i - i * (i + 1) // 2 + j - i - 1


def _trial_sparse(
    pw_json: str, variant: str, image_to_idx: dict[str, int], n_conds: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse one trial's pairwise_distances into (condensed_pair_indices, values)
    against the canonical n_conds-image ordering.

    Returns None for a trial with no parseable pairwise data (malformed/missing JSON --
    already warned about by the parser at load time).
    """
    pairwise = parse_pairwise_distances(pw_json)
    if not pairwise:
        return None
    canon = {src: image_to_idx[_curated_path_from_src(src, variant)] for pair in pairwise for src in pair}
    pair_i = np.empty(len(pairwise), dtype=np.int64)
    pair_j = np.empty(len(pairwise), dtype=np.int64)
    values = np.empty(len(pairwise), dtype=np.float64)
    for k, ((src_a, src_b), dist) in enumerate(pairwise.items()):
        a, b = canon[src_a], canon[src_b]
        pair_i[k], pair_j[k] = (a, b) if a < b else (b, a)
        values[k] = dist
    indices = _condensed_index(pair_i, pair_j, n_conds)
    return indices, values


def _ss_dense(v: np.ndarray) -> float:
    """Sum of squares over a (possibly NaN-containing) 1-D vector, NaN-safe."""
    return float(np.nansum(v ** 2))


def _scale_dense(v: np.ndarray) -> np.ndarray:
    """Divide a vector by the root sum of its own squares (NaN-safe); unchanged if norm is 0."""
    norm = np.sqrt(_ss_dense(v))
    return v / norm if norm > 0 else v


def _trial_weights(values: np.ndarray, method: str) -> np.ndarray:
    """Per-observation weights for one trial's raw (pre-alignment) values,
    matching rsatoolbox.rdm.combine._rescale()'s per-method weighting."""
    if method == "evidence":
        return (values ** 2).clip(0.2 ** 2)
    if method == "setsize":
        return np.full(values.shape, 1.0 / len(values))
    if method == "simple":
        return np.ones_like(values)
    raise ValueError(f"Unknown method: {method!r}")


def _sparse_combine(
    trials: list[tuple[np.ndarray, np.ndarray]],
    n_conds: int,
    method: str = "evidence",
    threshold: float = 1e-8,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine partial per-trial dissimilarity observations into one full RDM.

    Reimplements rsatoolbox.rdm.combine.rescale() + RDMs.mean(weights='rescalingWeights')
    (see module docstring for why) over sparse per-trial (indices, values) pairs instead
    of materializing a dense (n_trials, vector_len) array. Verified against rsatoolbox's
    own dense implementation in __tests__/test_behavioral.py.

    Returns (final_condensed, n_obs_per_pair) -- final_condensed is NaN at pair positions
    with zero total observations across every trial.
    """
    vector_len = n_conds * (n_conds - 1) // 2
    weights = [_trial_weights(values, method) for _, values in trials]
    row_ss = [_ss_dense(values) for _, values in trials]

    n_obs_per_pair = np.zeros(vector_len, dtype=np.int64)
    for indices, _ in trials:
        n_obs_per_pair[indices] += 1

    # Initial estimate: _scale(_mean(dissim)) with uniform (unweighted) averaging.
    sum0 = np.zeros(vector_len, dtype=np.float64)
    for indices, values in trials:
        sum0[indices] += values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # expected 0/0 -> NaN at uncovered pairs
        mean0 = np.where(n_obs_per_pair > 0, sum0 / np.maximum(n_obs_per_pair, 1), np.nan)
    current_estimate = _scale_dense(mean0)
    prev_estimate = np.full(vector_len, -np.inf)
    unscaled_mean = mean0

    n_iter = 0
    while _ss_dense(current_estimate - prev_estimate) > threshold:
        if n_iter >= max_iter:
            warnings.warn(
                f"_sparse_combine: did not converge to threshold={threshold} within "
                f"{max_iter} iterations; returning the current estimate.",
                RuntimeWarning, stacklevel=2,
            )
            break
        n_iter += 1
        prev_estimate = current_estimate.copy()

        weighted_sum = np.zeros(vector_len, dtype=np.float64)
        weight_sum = np.zeros(vector_len, dtype=np.float64)
        for (indices, values), w, ss in zip(trials, weights, row_ss):
            trial_est_sq_sum = np.nansum(prev_estimate[indices] ** 2)
            aligned_values = (values / np.sqrt(ss)) * np.sqrt(trial_est_sq_sum) if ss > 0 else np.zeros_like(values)
            weighted_sum[indices] += aligned_values * w
            weight_sum[indices] += w
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            unscaled_mean = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
        current_estimate = _scale_dense(unscaled_mean)

    return unscaled_mean, n_obs_per_pair


def _mean_combine(
    trials: list[tuple[np.ndarray, np.ndarray]], n_conds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine partial per-trial dissimilarity observations via a plain unweighted mean.

    Matches SpAM_Simulations' own combination scheme (mean_from_sum_and_count() in
    SpAM_Simulations/core/helpers.py) and the PiCS dataset's (Robbins et al., 2025,
    https://doi.org/10.3758/s13428-025-02732-0: "We computed a dissimilarity matrix
    based on the average Euclidean distance between each item and every other item"):
    total/count per pair, NaN if never observed -- no iterative rescaling, unlike
    _sparse_combine(). See module docstring for why both methods are built.

    Returns (final_condensed, n_obs_per_pair) -- final_condensed is NaN at pair positions
    with zero total observations across every trial.
    """
    vector_len = n_conds * (n_conds - 1) // 2
    total = np.zeros(vector_len, dtype=np.float64)
    n_obs_per_pair = np.zeros(vector_len, dtype=np.int64)
    for indices, values in trials:
        total[indices] += values
        n_obs_per_pair[indices] += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # expected 0/0 -> NaN at uncovered pairs
        condensed = np.where(n_obs_per_pair > 0, total / np.maximum(n_obs_per_pair, 1), np.nan)
    return condensed, n_obs_per_pair


_AGGREGATIONS = ("mean", "evidence", "simple")


def build_behavioral_rdm(
    variant: str,
    *,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    aggregation: Literal["mean", "evidence", "simple"] = "evidence",
) -> np.ndarray:
    """
    Build the behavioral (SpAM multi-arrangement) RDM for the given SHINE variant.

    Parameters
    ----------
    variant      : 'pre_shine' or 'post_shine'
    cache_dir    : directory containing the cached participants.parquet/trials.parquet
                   (see analysis/utils/load_data.py) -- NOT the raw data/ directory.
                   Defaults to the conventional in-repo location, which is only
                   populated at the main checkout (not copied into worktrees).
    aggregation  : how to combine each pair's multiple per-trial observations into
                   one value. One of:
                     'evidence' (default) -- iterative per-trial evidence-weighted
                         rescaling before averaging (Kriegeskorte & Mur, 2012,
                         https://doi.org/10.3389/fpsyg.2012.00245), as implemented in
                         rsatoolbox (van den Bosch et al., 2025,
                         https://doi.org/10.1101/2025.05.22.655542). See _sparse_combine().
                     'simple' -- same iterative rescaling as 'evidence', but with
                         uniform per-trial weighting instead of squared-distance
                         weighting (also from rsatoolbox.rdm.combine.rescale()). See
                         _sparse_combine().
                     'mean' -- plain unweighted average, no rescaling at all: the
                         scheme used by the PiCS dataset (Robbins et al., 2025,
                         https://doi.org/10.3758/s13428-025-02732-0) and by this
                         project's own SpAM_Simulations pipeline. See _mean_combine().

    Returns
    -------
    Condensed distance vector (float64, length 262_450). May contain NaN for image
    pairs never jointly observed in any trial -- see the printed coverage report.
    """
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant must be 'pre_shine' or 'post_shine', got {variant!r}")
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {_AGGREGATIONS}, got {aggregation!r}")

    data = load_data(cache_dir, prod_only=True)
    participants = _select_participants(data["participants"], variant)
    trials = data["trials"][
        data["trials"]["participant_id"].isin(participants["participant_id"])
        & ~data["trials"]["is_catch"]
    ]
    print(
        f"[behavioral] variant={variant!r}: {len(participants)} participants, "
        f"{len(trials)} non-catch trials (screening + experimental pooled)."
    )

    manifest = load_manifest()
    canonical_patterns = [p.replace("\\", "/") for p in manifest["curated_path"]]
    n_conds = len(canonical_patterns)
    image_to_idx = {img: i for i, img in enumerate(canonical_patterns)}

    trial_data = [
        t for t in (
            _trial_sparse(pw, variant, image_to_idx, n_conds)
            for pw in trials["pairwise_distances"]
        ) if t is not None
    ]
    print(f"[behavioral] Built {len(trial_data)} per-trial partial observation sets.")

    if aggregation == "mean":
        condensed, n_obs_per_pair = _mean_combine(trial_data, n_conds)
    else:
        condensed, n_obs_per_pair = _sparse_combine(trial_data, n_conds, method=aggregation)

    n_uncovered = int(np.sum(n_obs_per_pair == 0))
    pct_covered = 100.0 * (1 - n_uncovered / n_obs_per_pair.size)
    print(
        f"[behavioral] Pair coverage: {pct_covered:.2f}% covered "
        f"({n_uncovered} of {n_obs_per_pair.size} pairs never jointly observed). "
        f"Judgements/pair (covered pairs only): mean={n_obs_per_pair[n_obs_per_pair > 0].mean():.2f}, "
        f"sd={n_obs_per_pair[n_obs_per_pair > 0].std(ddof=1):.2f}."
    )
    if n_uncovered > 0:
        print(
            f"[behavioral] WARNING: {n_uncovered} pairs have zero observations and will be "
            f"NaN in the saved RDM -- this is a real coverage gap, not silently interpolated."
        )

    short = f"behav_{'pre' if variant == 'pre_shine' else 'post'}_{aggregation}"
    metric = (
        "spam_plain_mean" if aggregation == "mean"
        else f"spam_multi_arrangement_{aggregation}"
    )
    save_rdm(
        short,
        condensed,
        metric=metric,
        source="analysis.rdms.behavioral",
        extra={
            "variant": variant,
            "n_participants": int(len(participants)),
            "n_trials": int(len(trial_data)),
            "aggregation": aggregation,
            "pct_pairs_covered": pct_covered,
            "mean_judgements_per_covered_pair": float(n_obs_per_pair[n_obs_per_pair > 0].mean()),
            "sd_judgements_per_covered_pair": float(n_obs_per_pair[n_obs_per_pair > 0].std(ddof=1)),
        },
    )
    print(f"[behavioral] Saved D_{short}.npy  (length {len(condensed)})")
    return condensed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-dir", default=_DEFAULT_CACHE_DIR,
        help="Directory with participants.parquet/trials.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--aggregation", nargs="+", default=list(_AGGREGATIONS), choices=list(_AGGREGATIONS),
        help="One or more aggregation methods to build (default: both of %(choices)s)",
    )
    parser.add_argument("--only", nargs="+", choices=["pre_shine", "post_shine"], default=["pre_shine", "post_shine"])
    args = parser.parse_args()
    for variant in args.only:
        for aggregation in args.aggregation:
            build_behavioral_rdm(variant, cache_dir=args.cache_dir, aggregation=aggregation)


if __name__ == "__main__":
    main()
