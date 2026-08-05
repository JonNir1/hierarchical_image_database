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
from typing import Dict, List, Sequence

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


def gradient_is_monotone(table: pd.DataFrame) -> bool:
    """Do mean distances fall monotonically as pairs become more closely related?"""
    means = table.sort_values("level")["mean_distance"].to_numpy()
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
    return {
        "distribution": distribution_comparison(sim_condensed, pilot_condensed),
        "gradient": merged,
        "sim_gradient_monotone": gradient_is_monotone(sim_grad),
        "pilot_gradient_monotone": gradient_is_monotone(pilot_grad),
        "max_abs_std_gap_diff": float(merged["std_gap_diff"].abs().max()),
    }
