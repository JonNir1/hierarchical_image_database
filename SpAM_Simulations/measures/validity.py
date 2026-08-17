"""Is a simulated cohort empirically realistic?

Simulated subjects are generated from a ground truth that was itself fitted to the pilot, so the
obvious check - do simulated distances look like real ones? - is close to guaranteed to pass by
construction. It is worth running as a floor (a mismatch would expose mis-scaled noise or a wrong
noise family) but it is weak evidence: a distance histogram is a univariate marginal, and very
different geometries produce near-identical ones. It says nothing about *which* pairs are close,
which is the thing the downstream stimulus construction depends on.

So this module pairs it with a structural check that was never fitted to. The manifest relpaths
encode the full semantic hierarchy (``animate/animal/body/bird/chick1.png``), and real perceptual
data should place same-subcategory pairs closer than same-category pairs, and those closer than
cross-category pairs. Whether a simulated cohort reproduces that gradient is not something the
noise-model fit could have arranged, and it is directly about near-neighbour structure.

**Scale.** ``task_v3_experiment._simulate_trial`` scales the *noise* by each trial's arrangement
spread but does not normalise the distances, so simulated distances live in ground-truth embedding
units while pilot distances are canvas-diagonal-normalised to [0, 1]. A raw Wasserstein between them
is meaningless. Everything here is compared after median-rescaling, and the gradient is expressed as
standardised gaps, which are scale-free.

Uses ``SpAM_Task/stimuli_manifest.json`` as the only ordering source. ``images/manifest.csv`` (what
``analysis/rdms/common.py`` reads) is not usable: it has 725 rows but only 719 distinct paths, with
six caucasian faces duplicated and the six hispanic ones missing entirely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# Level names by how deep the lowest common ancestor sits. Index 0 is "no shared prefix", which
# cannot occur here because every path starts animate/ or inanimate/.
LEVEL_NAMES = ("unrelated", "cross_category", "same_category", "same_subcategory", "same_leaf")


def _dir_parts(relpath: str) -> tuple:
    """Directory segments of a manifest relpath, excluding the filename."""
    return Path(str(relpath).replace("\\", "/")).parts[:-1]


def _lca_depth(a: tuple, b: tuple) -> int:
    """Number of matching leading segments, i.e. the depth of the lowest common ancestor."""
    depth = 0
    for x, y in zip(a, b):
        if x != y:
            break
        depth += 1
    return depth


def hierarchy_levels(manifest_images: Sequence[str]) -> np.ndarray:
    """Condensed vector of LCA depth for every image pair, in ``scipy`` condensed order.

    Deliberately a small reimplementation of ``analysis.rdms.semantic_km``'s ``_dir_parts`` /
    ``_lca_depth`` rather than an import: that module pulls in ``analysis.rdms.common``, which reads
    the broken ``images/manifest.csv`` and would need a third sparse-checkout path on EC2. The
    tests assert the two agree on a hand-written path list.
    """
    parts = [_dir_parts(p) for p in manifest_images]
    n = len(parts)
    out = np.empty(n * (n - 1) // 2, dtype=np.int16)
    idx = 0
    for i in range(n - 1):
        pi = parts[i]
        for j in range(i + 1, n):
            out[idx] = _lca_depth(pi, parts[j])
            idx += 1
    return out


def gradient_table(condensed: np.ndarray, levels: np.ndarray) -> pd.DataFrame:
    """Mean distance per hierarchy level, plus each level's gap from the shallowest, standardised.

    ``std_gap`` is ``(mean_at_level_0 - mean_at_level) / sd_overall``, so it is invariant to the
    arbitrary scale of MDS output and comparable between a simulated cohort and the real pilot.
    A well-behaved perceptual space gives gaps that increase monotonically with depth.
    """
    condensed = np.asarray(condensed, dtype=np.float64)
    ok = np.isfinite(condensed)
    sd = float(np.nanstd(condensed[ok])) if ok.any() else np.nan
    base = np.nan
    rows: List[Dict[str, float]] = []
    for lvl in sorted(np.unique(levels)):
        sel = ok & (levels == lvl)
        vals = condensed[sel]
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        if np.isnan(base):
            base = mean
        rows.append({
            "level": int(lvl),
            "level_name": LEVEL_NAMES[int(lvl)] if int(lvl) < len(LEVEL_NAMES) else f"depth_{lvl}",
            "n_pairs": int(vals.size),
            "mean_distance": mean,
            "sd_distance": float(vals.std()),
            "std_gap": float((base - mean) / sd) if sd and np.isfinite(sd) else np.nan,
        })
    return pd.DataFrame(rows)


# A hierarchy level is only worth testing for monotonicity if enough pairs sit in it. The deepest
# level of this stimulus set (`depth_5`) holds 348 of 262,450 pairs, of which the GT's own subjects
# judged 100; the empirical ordering it is asked to reproduce is a same_leaf/depth_5 gap of 0.007
# measured on 121 pilot pairs. A GT fitted on that cell is mostly interpolating, so requiring the
# embedding to reproduce its ordering is requiring it to reproduce noise - and because a single
# inversion flips the whole boolean, an underpowered cell silently condemns an otherwise sound run.
#
# 500 is a judgement call, not a derived quantity: it is the round number that drops `depth_5` while
# keeping `same_leaf` (4,758 pairs simulated / 1,723 pilot). Levels excluded by it are reported
# rather than hidden, so the choice stays visible wherever the flag is used.
MIN_GRADIENT_PAIRS = 500


def gradient_is_monotone(table: pd.DataFrame, min_pairs: int = 0) -> bool:
    """Do mean distances fall monotonically as pairs become more closely related?

    ``min_pairs`` drops levels with too little support before testing. It defaults to 0 - the strict
    all-levels test - so this primitive keeps saying exactly what it says; :func:`compare_to_pilot`
    is where :data:`MIN_GRADIENT_PAIRS` is applied. Fewer than two surviving levels is vacuously
    monotone, so callers that care should check how many levels were actually tested.
    """
    kept = table[table["n_pairs"] >= min_pairs] if min_pairs > 0 else table
    means = kept.sort_values("level")["mean_distance"].to_numpy()
    return bool(np.all(np.diff(means) <= 0))


def distribution_comparison(sim_condensed: np.ndarray, pilot_condensed: np.ndarray,
                            rescale: str = "median") -> Dict[str, float]:
    """Compare two distance distributions, after putting them on a common scale.

    ``rescale="median"`` divides each by its own median, which is the minimum needed to make the
    comparison meaningful at all given that simulated and pilot distances are in different units.
    ``rescale="none"`` skips it, for the case where both are already normalised.
    """
    sim = np.asarray(sim_condensed, dtype=np.float64)
    pilot = np.asarray(pilot_condensed, dtype=np.float64)
    sim = sim[np.isfinite(sim)]
    pilot = pilot[np.isfinite(pilot)]
    if sim.size == 0 or pilot.size == 0:
        raise ValueError("both distributions must have at least one finite value")
    if rescale == "median":
        sim = sim / np.median(sim)
        pilot = pilot / np.median(pilot)
    elif rescale != "none":
        raise ValueError(f"rescale must be 'median' or 'none', got {rescale!r}")

    pct = [1, 5, 25, 50, 75, 95, 99]
    out = {
        "wasserstein": float(wasserstein_distance(sim, pilot)),
        "sim_mean": float(sim.mean()), "pilot_mean": float(pilot.mean()),
        # CV is the shape statistic that survives rescaling, so it carries the real signal here.
        "sim_cv": float(sim.std() / sim.mean()), "pilot_cv": float(pilot.std() / pilot.mean()),
        "n_sim": int(sim.size), "n_pilot": int(pilot.size),
    }
    for p in pct:
        out[f"sim_p{p}"] = float(np.percentile(sim, p))
        out[f"pilot_p{p}"] = float(np.percentile(pilot, p))
    out["cv_ratio"] = out["sim_cv"] / out["pilot_cv"] if out["pilot_cv"] else np.nan
    return out


def compare_to_pilot(sim_condensed: np.ndarray, pilot_condensed: np.ndarray,
                     manifest_images: Sequence[str]) -> Dict[str, object]:
    """Both checks at once: the distance distribution, and the semantic gradient.

    ``pilot_condensed`` should be the pooled aggregate (``gt_construction.aggregate_subjects``'s
    first return, with unobserved pairs set to NaN) so that only judged pairs contribute.
    """
    levels = hierarchy_levels(manifest_images)
    sim_grad = gradient_table(sim_condensed, levels)
    pilot_grad = gradient_table(pilot_condensed, levels)
    merged = sim_grad.merge(pilot_grad, on=["level", "level_name"], suffixes=("_sim", "_pilot"))
    merged["std_gap_diff"] = merged["std_gap_sim"] - merged["std_gap_pilot"]
    # Test the SAME levels on both sides. Applying the support threshold to each table separately
    # would compare a sim gradient over one level set against a pilot gradient over another, since
    # the two have different pair counts at the same level.
    supported = merged.loc[(merged["n_pairs_sim"] >= MIN_GRADIENT_PAIRS)
                           & (merged["n_pairs_pilot"] >= MIN_GRADIENT_PAIRS), "level"]
    tested = sorted(int(x) for x in supported)
    skipped = sorted(int(x) for x in merged["level"] if int(x) not in tested)
    sim_tested = sim_grad[sim_grad["level"].isin(tested)]
    pilot_tested = pilot_grad[pilot_grad["level"].isin(tested)]
    return {
        "distribution": distribution_comparison(sim_condensed, pilot_condensed),
        "gradient": merged,
        "sim_gradient_monotone": gradient_is_monotone(sim_tested),
        "pilot_gradient_monotone": gradient_is_monotone(pilot_tested),
        "gradient_levels_tested": tested,
        "gradient_levels_skipped": skipped,
        "sim_gradient_monotone_all_levels": gradient_is_monotone(sim_grad),
        "max_abs_std_gap_diff": float(merged["std_gap_diff"].abs().max()),
    }


def summarise_gradients(per_cell: pd.DataFrame, group: str = "arm") -> pd.DataFrame:
    """Aggregate per-cell gradient outcomes into one row per arm.

    Scoring a single cell answers "is this configuration realistic?" and then reports it as though
    it answered "are the cohorts realistic?". Those differ: the sweep varies softness, screening
    threshold and dispersion, and a gradient that survives one combination need not survive all of
    them. Worse, the single cell was whichever came first in dict order, so the answer depended on
    insertion order rather than on anything about the model.

    ``monotone_frac`` is the number to read. 1.0 says every configuration reproduced the ordering;
    0.0 says none did and the model is wrong; anything between says the gradient depends on levers
    the sweep is varying, which is itself the finding and is invisible to a boolean.
    """
    if per_cell.empty:
        return pd.DataFrame(columns=[group, "n_cells", "monotone_frac", "n_monotone"])
    rows = []
    for name, sub in per_cell.groupby(group, sort=True):
        row = {
            group: name,
            "n_cells": int(len(sub)),
            "n_monotone": int(sub["monotone"].sum()),
            "monotone_frac": float(sub["monotone"].mean()),
        }
        if "max_abs_std_gap_diff" in sub:
            gaps = sub["max_abs_std_gap_diff"].astype(float)
            row.update({
                "gap_mean": float(gaps.mean()), "gap_sd": float(gaps.std(ddof=0)),
                "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- noise vs distance
# The third validity check, and the sharpest of the three, because it is about the shape of the
# NOISE rather than the shape of the signal.
#
# The empirical finding (analysis/pilot/figures.py, "Reliability vs. distance"): for every image
# pair a subject judged twice, plot the RMSE between the two judgements against their mean, and the
# curve is an inverted U. Pairs that are clearly similar and pairs that are clearly dissimilar are
# both judged consistently; the ambiguous middle is where subjects disagree with themselves. A
# simulation that does not reproduce that is generating the wrong kind of noise, however well its
# distance histogram matches.
#
# **The two ends are not equally strong tests.** The low-distance rise is close to forced:
# distances cannot go below zero, so as the true separation approaches 0 the gap between two noisy
# realisations is squeezed against that floor, and any additive-noise model reproduces it. The
# high-distance *turnover* is the discriminating one, and it needs a bounded canvas: a pair already
# at opposite corners cannot move much further apart, so its upper tail is truncated.
#
# Measured on the real pilot (47 subjects, 12,540 repeat pairs): the curve rises off a floor
# (`rise_from_first` 0.83) to a LATE peak (`peak_bin_frac` 0.78) and then falls sharply
# (`drop_from_peak` 0.37). The turnover is confined to the top bin, which is why this module
# describes the curve relative to its peak rather than by comparing thirds - see
# :func:`noise_curve_shape`.
TURNOVER_TOLERANCE = 0.10   # a rise or drop counts only above this fraction of the peak


def repeat_pairs(subjects: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Pool every subject's ``(original, repeat)`` trial distance vectors into two flat arrays.

    Reads ``PilotSubject.retest_pairs``, which the loader already populates, so this needs neither
    the trial-level parser nor a second pass over the raw CSVs.
    """
    orig: List[np.ndarray] = []
    repeat: List[np.ndarray] = []
    for s in subjects:
        for o, r in getattr(s, "retest_pairs", []):
            o = np.asarray(o, dtype=np.float64)
            r = np.asarray(r, dtype=np.float64)
            ok = np.isfinite(o) & np.isfinite(r)
            if ok.any():
                orig.append(o[ok])
                repeat.append(r[ok])
    if not orig:
        return np.array([]), np.array([])
    return np.concatenate(orig), np.concatenate(repeat)


def simulate_repeat_pairs(gt_embeddings: np.ndarray, subjects_noise_scale: float,
                          n_subjects: int = 40, trials_per_subject: int = 4,
                          images_per_trial: int = 20, perspective_dispersion: float = 0.3,
                          noise_df: int = 5, lognormal_sigma: float = 0.0,
                          seed: int = 0,
                          trial_simulator=None) -> Tuple[np.ndarray, np.ndarray]:
    """``(d_orig, d_repeat)`` from verbatim repeat trials of the **deployed generative model**.

    Calls the trial simulator directly rather than re-deriving the projection, so the curve
    describes the model the sweep actually runs. A repeat re-projects the same items with the same
    perspective weights and differs only by fresh placement noise, exactly as
    ``simulate_task_v3_experiment`` does it - the difference is only that the per-pair values are
    kept here instead of being collapsed to a Spearman.

    ``trial_simulator`` is the same seam ``task_v4_experiment`` carries, and passing it is what
    makes this check meaningful under task-v5: the turnover this function measures is *caused* by
    the bounded canvas, so measuring it against the unbounded v3 default would report on a model
    the sweep does not run. It defaults to v3's ``_simulate_trial`` only so the v3/v4 callers and
    their bit-exactness tests are unaffected.

    Standalone rather than plumbed through the sweep, and deliberately so: this measures a property
    of the **noise model**, not of any particular cohort size or allocation arm, so it needs no MDS
    and costs seconds. The accumulator arrays are throwaway.
    """
    from SpAM_Simulations.models.noise_population import draw_subject_noises, resolve_family
    from SpAM_Simulations.models.task_v3_experiment import _draw_perspective_weights, _simulate_trial

    simulate = _simulate_trial if trial_simulator is None else trial_simulator
    gt = np.asarray(gt_embeddings, dtype=np.float32)
    n_images, n_dims = gt.shape
    if images_per_trial > n_images:
        raise ValueError(f"images_per_trial={images_per_trial} exceeds n_images={n_images}")
    rng = np.random.default_rng(seed)
    pair_rows, pair_cols = np.triu_indices(images_per_trial, k=1)
    n_pairs = n_images * (n_images - 1) // 2
    # The v4 draw, which handles both the |t(df)| and lognormal families, so a curve measured here
    # describes the same noise population the calibrated sweep runs.
    family, shape = resolve_family(noise_df, lognormal_sigma)
    scales = draw_subject_noises(n_subjects, subjects_noise_scale, rng=rng,
                                 family=family, shape=shape)

    orig: List[np.ndarray] = []
    repeat: List[np.ndarray] = []
    for subject_noise in scales:
        weights = _draw_perspective_weights(n_dims, perspective_dispersion, rng)
        for _ in range(trials_per_subject):
            trial = rng.choice(n_images, size=images_per_trial, replace=False)
            obs, n_obs = np.zeros(n_pairs, dtype=np.float64), np.zeros(n_pairs, dtype=np.int32)
            _, d1, _ = simulate(trial, pair_rows, pair_cols, n_images, gt, weights,
                                float(subject_noise), obs, n_obs, rng)
            _, d2, _ = simulate(trial, pair_rows, pair_cols, n_images, gt, weights,
                                float(subject_noise), obs, n_obs, rng)
            orig.append(np.asarray(d1, dtype=np.float64))
            repeat.append(np.asarray(d2, dtype=np.float64))
    return np.concatenate(orig), np.concatenate(repeat)


def noise_vs_distance(d_orig: np.ndarray, d_repeat: np.ndarray, n_bins: int = 10,
                      binning: str = "quantile", rescale: str = "median") -> pd.DataFrame:
    """Binned RMSE between two judgements of the same pair, against how far apart they put it.

    One row per bin of ``pair_mean = (d_orig + d_repeat) / 2``, carrying
    ``rmse = sqrt(mean((d_orig - d_repeat)**2))``. This is the same quantity
    ``analysis/pilot/figures.py`` plots, and ``binning="fixed"`` with ``rescale="none"`` reproduces
    its fixed-width bins; a test asserts the two agree on the same input.

    ``binning="quantile"`` is the default here because this comparison is cross-scale. Simulated
    distances are in ground-truth embedding units while pilot distances are canvas-diagonal
    normalised, so fixed-width bins would not describe the same parts of the two distributions.
    Equal-count bins also stabilise the RMSE estimate at the sparse extremes, which is exactly where
    the shape is being read.

    ``rescale="median"`` divides both arrays by the pooled median first, putting the axes in
    median-distance units. **Under task-v5 this is no longer needed for comparability**: the canvas
    simulator divides every simulated distance by the canvas diagonal
    (:func:`SpAM_Simulations.models.canvas.canvas_distances`), exactly as the deployed task does, so
    both sources already live on the same [0, 1] scale. It is kept as the default only so the
    v3/v4 callers and their bit-exactness tests are unaffected; pass ``rescale="none"`` to read the
    curve in native canvas-diagonal units, which is what the v5 report does.

    The SEM uses the pilot figure's delta method: ``SEM(sqrt(X)) ~= SEM(X) / (2 * sqrt(mean(X)))``.
    It describes how well each bin's RMSE is pinned down by the pairs in it, which is a different
    question from how much the curve varies between cohorts; :func:`noise_vs_distance_draws`
    answers that one.
    """
    o = np.asarray(d_orig, dtype=np.float64)
    r = np.asarray(d_repeat, dtype=np.float64)
    if o.shape != r.shape:
        raise ValueError(f"arrays must match in length, got {o.shape} and {r.shape}")
    ok = np.isfinite(o) & np.isfinite(r)
    o, r = o[ok], r[ok]
    if o.size == 0:
        raise ValueError("no finite (original, repeat) pairs to bin")
    if rescale == "median":
        med = float(np.median(np.concatenate([o, r])))
        if med > 0:
            o, r = o / med, r / med
    elif rescale != "none":
        raise ValueError(f"rescale must be 'median' or 'none', got {rescale!r}")

    pair_mean = 0.5 * (o + r)
    sq_diff = (o - r) ** 2
    if binning == "quantile":
        edges = np.unique(np.quantile(pair_mean, np.linspace(0, 1, n_bins + 1)))
    elif binning == "fixed":
        edges = np.linspace(pair_mean.min(), pair_mean.max(), n_bins + 1)
    else:
        raise ValueError(f"binning must be 'quantile' or 'fixed', got {binning!r}")
    if edges.size < 2:
        raise ValueError("pair_mean is constant; no bins can be formed")

    idx = np.clip(np.digitize(pair_mean, edges[1:-1], right=False), 0, len(edges) - 2)
    rows: List[Dict[str, float]] = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        rmse = float(np.sqrt(sq_diff[sel].mean()))
        # Delta method, matching the pilot figure. Degenerate at rmse == 0, where the SEM is 0 too.
        sem_sq = float(sq_diff[sel].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        sem_rmse = sem_sq / (2 * rmse) if rmse > 0 and np.isfinite(sem_sq) else 0.0
        rows.append({
            "bin": b, "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
            "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
            "mean_pair_distance": float(pair_mean[sel].mean()),
            "n_pairs": n, "rmse": rmse, "sem_rmse": float(sem_rmse),
        })
    return pd.DataFrame(rows)


def noise_curve_shape(table: pd.DataFrame, tolerance: float = TURNOVER_TOLERANCE
                      ) -> Dict[str, float]:
    """Scale-free descriptors of a :func:`noise_vs_distance` curve.

    **Peak-relative, not thirds-relative, and the difference matters.** An earlier version of this
    function split the bins into low/middle/high thirds and asked whether each flank was quieter
    than the middle. Measured against the real pilot that summary returns ``high_over_mid = 1.30``
    and so calls the empirical data *not* an inverted U - because the turnover is confined to the
    **top bin** and the peak sits at 78% along the range, not at 50%. The thirds average the drop
    away. The curve is a long rise to a late peak followed by a sharp fall, so it is described that
    way:

    * ``rise_from_first`` - ``1 - rmse[0] / rmse[peak]``, how far the curve climbs off its low-end
      floor. Pilot: 0.83.
    * ``drop_from_peak`` - ``1 - rmse[-1] / rmse[peak]``, how far it falls after the peak. This is
      the discriminating quantity. Pilot: 0.37; an unbounded generative model gives exactly 0.
    * ``peak_bin_frac`` - where the noisiest bin sits along the range. Pilot: 0.78.

    The thirds are still returned as secondary description, but no verdict rests on them.
    """
    if table.empty:
        raise ValueError("cannot describe the shape of an empty curve")
    rmse = table["rmse"].to_numpy(dtype=float)
    counts = table["n_pairs"].to_numpy(dtype=float)
    peak = int(np.argmax(rmse))
    peak_rmse = float(rmse[peak])
    rise = 1.0 - rmse[0] / peak_rmse if peak_rmse > 0 else np.nan
    drop = 1.0 - rmse[-1] / peak_rmse if peak_rmse > 0 else np.nan

    cut = max(1, len(rmse) // 3)

    def weighted(lo, hi) -> float:
        seg_r, seg_n = rmse[lo:hi], counts[lo:hi]
        return float(np.average(seg_r, weights=seg_n)) if seg_n.sum() else np.nan

    low, mid, high = weighted(0, cut), weighted(cut, len(rmse) - cut), weighted(len(rmse) - cut, None)
    return {
        "rise_from_first": float(rise), "drop_from_peak": float(drop),
        "peak_bin_frac": peak / (len(rmse) - 1) if len(rmse) > 1 else np.nan,
        "rmse_peak": peak_rmse, "rmse_first": float(rmse[0]), "rmse_last": float(rmse[-1]),
        # Secondary, descriptive only - see the docstring for why no verdict uses these.
        "rmse_low": low, "rmse_mid": mid, "rmse_high": high,
        "has_low_floor": bool(np.isfinite(rise) and rise > tolerance),
        "turns_over": bool(np.isfinite(drop) and drop > tolerance),
        "is_inverted_u": bool(np.isfinite(rise) and np.isfinite(drop)
                              and rise > tolerance and drop > tolerance),
        "n_bins": int(len(rmse)),
    }


def null_distances(num_dots: int = 20, num_trials: int = 2000, seed: int = 42) -> np.ndarray:
    """Pairwise distances for ``num_dots`` points dropped uniformly on the unit square.

    Delegates to ``analysis/pilot/simulate_null_distances.py`` rather than restating it, so the
    report's null and the pilot analysis's null cannot drift apart. Distances come back divided by
    the unit-square diagonal, which is the same normalisation the deployed task and the v5 canvas
    simulator apply, so the three sources are directly comparable with no rescaling.
    """
    from analysis.pilot.simulate_null_distances import simulate
    return simulate(num_dots, num_trials, seed=seed)


def null_distance_summary(sim_distances: np.ndarray, pilot_distances: np.ndarray,
                          num_dots: int = 20, num_trials: int = 2000,
                          seed: int = 42) -> pd.DataFrame:
    """Mean, SD and quartiles of the three distance distributions, on one common axis.

    Random placement is the floor this task has to clear: if participants were dropping images
    without regard to similarity, their distances would look like the null. The gap between the null
    and the two participant rows is therefore a crude but assumption-free check that the arrangement
    carries structure at all, independent of any ground truth.
    """
    null = null_distances(num_dots=num_dots, num_trials=num_trials, seed=seed)
    rows = []
    for name, values in (("random placement (null)", null),
                         ("simulated participants", np.asarray(sim_distances, dtype=np.float64)),
                         ("pilot participants", np.asarray(pilot_distances, dtype=np.float64))):
        v = values[np.isfinite(values)]
        rows.append({"source": name, "n": int(v.size), "mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)), "median": float(np.median(v)),
                     "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))})
    return pd.DataFrame(rows)


def noise_vs_distance_draws(simulate_pairs, n_draws: int = 20, base_seed: int = 0,
                            **kwargs) -> pd.DataFrame:
    """The binned RMSE curve over ``n_draws`` independent simulated cohorts.

    ``simulate_pairs(seed)`` returns one cohort's ``(d_orig, d_repeat)``. Returns the per-bin mean
    and **SD across cohorts**, which is the spread the rest of the report shows. The within-bin SEM
    that :func:`noise_vs_distance` reports is a precision, not a spread, and with millions of pairs
    per bin it is invisible.
    """
    frames = [noise_vs_distance(*simulate_pairs(base_seed + i), **kwargs).assign(draw=i)
              for i in range(n_draws)]
    stacked = pd.concat(frames, ignore_index=True)
    agg = stacked.groupby("bin").agg(
        bin_center=("bin_center", "mean"), mean_pair_distance=("mean_pair_distance", "mean"),
        n_pairs=("n_pairs", "mean"), rmse=("rmse", "mean"), sd_rmse=("rmse", "std"),
        n_draws=("rmse", "size")).reset_index()
    return agg


def compare_noise_vs_distance(sim_pairs: Tuple[np.ndarray, np.ndarray],
                              pilot_pairs: Tuple[np.ndarray, np.ndarray],
                              n_bins: int = 10) -> Dict[str, object]:
    """Both curves and both shape summaries, side by side.

    ``*_pairs`` are ``(d_orig, d_repeat)``, from :func:`repeat_pairs` for the pilot and from a
    repeat-trial simulation for the model. Returns the stacked per-bin table plus a ``shape`` frame
    with one row per source, so the two ends can be read against each other directly.

    ``turnover_matches`` is the comparison that carries information: the low-end rise is near-forced
    for any additive-noise model, so agreeing there says little, while the turnover depends on a
    bounded canvas.
    """
    sim = noise_vs_distance(*sim_pairs, n_bins=n_bins)
    pilot = noise_vs_distance(*pilot_pairs, n_bins=n_bins)
    shapes = pd.DataFrame([{"source": "sim", **noise_curve_shape(sim)},
                           {"source": "pilot", **noise_curve_shape(pilot)}])
    curves = pd.concat([sim.assign(source="sim"), pilot.assign(source="pilot")], ignore_index=True)
    by_source = shapes.set_index("source")
    return {
        "curves": curves,
        "shape": shapes,
        "sim_is_inverted_u": bool(by_source.loc["sim", "is_inverted_u"]),
        "pilot_is_inverted_u": bool(by_source.loc["pilot", "is_inverted_u"]),
        "low_end_matches": bool(by_source.loc["sim", "has_low_floor"]
                                == by_source.loc["pilot", "has_low_floor"]),
        "turnover_matches": bool(by_source.loc["sim", "turns_over"]
                                 == by_source.loc["pilot", "turns_over"]),
        # How far short the model falls on the discriminating quantity, as a plain difference.
        "drop_gap": float(by_source.loc["pilot", "drop_from_peak"]
                          - by_source.loc["sim", "drop_from_peak"]),
    }
