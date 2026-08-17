"""Is a fitted ground truth a faithful summary of the data it was fitted on?

The stage-2 validity block only ever sees *simulated cohort* distances, so when its semantic-gradient
check fails it cannot say whether the GT lacks the structure or the noise model destroyed it. These
diagnostics answer that directly: they compare the GT against the raw aggregate it was fitted from,
with no simulation anywhere in the path. They were written as throwaway scripts while debugging a
task-v5 stage-2 run and are kept here because the questions recur on every new GT.

What the four tables are for, in the order you should read them:

``level_coverage``
    How many pairs at each semantic level anyone actually judged. A level the subjects barely
    observed is one the embedding mostly *interpolates*, and asking the GT to reproduce its ordering
    asks it to reproduce noise. On the pre-SHINE pilot set, ``depth_5`` holds 348 pairs of which 100
    were judged - which is why :data:`validity.MIN_GRADIENT_PAIRS` exists.

``gradient``
    :func:`validity.gradient_table` applied to ``pdist(GT)``. Non-monotonicity here is inherited by
    every cohort ever drawn from this GT, so no change to the noise model, canvas, or dispersion can
    fix it.

``gt_vs_raw``
    Within each level, Spearman between the GT's distances and the raw aggregate's, over pairs the
    subjects judged. **This is an IN-SAMPLE fit statistic, not a validity statistic**: the GT was
    fitted on the same subjects the aggregate pools, and a 725xD embedding has enough freedom to
    chase noise in a 31%-observed matrix. On its own it cannot separate *compression* (means
    squeezed, per-pair order intact) from *information loss* (order destroyed) - it is only
    interpretable next to ``noise_ceiling``.

``noise_ceiling``
    The control ``gt_vs_raw`` requires. A within-level correlation is attenuated by restricted range
    and by the raw aggregate's own unreliability, so a low value is not by itself evidence the
    embedding discarded anything - and a HIGH one is not evidence it preserved anything.
    ``frac_of_ceiling`` is the number to read. Near 1 means the GT extracted what the data supports.
    Comfortably above 1 means the GT is reproducing variance the data cannot reproduce in itself,
    i.e. it is fitting noise, and any within-level structure it appears to carry is an artefact.

    On the 41-subject pre-SHINE pilot this comes out damning at every level except
    ``same_subcategory``: ceilings of 0.07 / 0.02 / 0.13 / 0.41 / 0.05 against GT correlations of
    0.44 / 0.45 / 0.50 / 0.49 / 0.13, i.e. ``frac_of_ceiling`` of 5.9, 20.7, 3.9, 1.2, 2.8. The
    honest reading is that this data supports almost no *within-level* per-pair structure, which is
    consistent with stage 1's out-of-sample split-half Spearman peaking at 0.233. The GT's apparent
    within-level agreement is mostly in-sample fit.

    The clean version of this test embeds one half and scores it against the other half's aggregate.
    That is exactly ``gt_construction.cross_validate_ndim``, which needs R and an MDS fit per split;
    it is not duplicated here. These tables are the cheap screen that tells you whether to go run it.

The splits here are plain random halves, deliberately unlike :func:`gt_construction.draw_valid_splits`
which discards halves whose observed-pair graph is disconnected. That filter exists because MDS
cannot run on a disconnected graph; no MDS runs here, so applying it would only import its selection
bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from SpAM_Simulations.empirical.gt_construction import aggregate_subjects, observed_mask
from SpAM_Simulations.measures.validity import LEVEL_NAMES, gradient_table, hierarchy_levels

DEFAULT_N_SPLITS = 20


def _level_name(level: int) -> str:
    return LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f"depth_{level}"


def level_coverage(subjects: Sequence, levels: np.ndarray) -> pd.DataFrame:
    """Per semantic level: how many pairs exist, and how many anyone judged."""
    observed = np.asarray(observed_mask(subjects), dtype=float) > 0
    rows = []
    for level in sorted(int(x) for x in np.unique(levels)):
        mask = levels == level
        rows.append({
            "level": level,
            "level_name": _level_name(level),
            "n_pairs": int(mask.sum()),
            "n_observed": int(observed[mask].sum()),
            "observed_frac": float(observed[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def gt_gradient(coords: np.ndarray, levels: np.ndarray) -> pd.DataFrame:
    """The semantic gradient of the ground truth itself, with no simulation in the path."""
    return gradient_table(pdist(np.asarray(coords, dtype=np.float64)), levels)


def gt_vs_raw(coords: np.ndarray, subjects: Sequence, levels: np.ndarray) -> pd.DataFrame:
    """Within-level Spearman between GT distances and the raw aggregate, over judged pairs only.

    Unobserved pairs are excluded because the GT's value there is interpolation, so including them
    would score the embedding against a number no subject produced.

    **In-sample.** The GT is fitted on these very subjects, so this measures fit, not recovery. It
    means nothing without :func:`raw_noise_ceiling` beside it; :func:`diagnose` joins the two and
    derives ``frac_of_ceiling`` precisely so the raw number is never read alone.
    """
    mean, _ = aggregate_subjects(subjects)
    observed = np.asarray(observed_mask(subjects), dtype=float) > 0
    gt = pdist(np.asarray(coords, dtype=np.float64))
    rows = []
    for level in sorted(int(x) for x in np.unique(levels)):
        mask = (levels == level) & observed
        if mask.sum() < 3:
            continue
        rho, p = spearmanr(gt[mask], mean[mask])
        rows.append({
            "level": level,
            "level_name": _level_name(level),
            "n_observed": int(mask.sum()),
            "spearman": float(rho),
            "p_value": float(p),
            "raw_sd": float(np.std(mean[mask])),
            "gt_sd": float(np.std(gt[mask])),
        })
    return pd.DataFrame(rows)


def raw_noise_ceiling(subjects: Sequence, levels: np.ndarray, n_splits: int = DEFAULT_N_SPLITS,
                      rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """How well the raw data agrees with itself at each level: the ceiling ``gt_vs_raw`` can reach.

    Each draw splits subjects into disjoint halves, pools each, and correlates them over the pairs
    *both* halves observed. ``ceiling_half`` is that correlation; ``ceiling_full`` applies the
    Spearman-Brown correction ``2r / (1 + r)`` to project it to full sample size, which is the right
    comparison for ``gt_vs_raw`` because the GT is fitted on all subjects, not half of them.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(subjects)
    if n < 4:
        raise ValueError(f"need at least 4 subjects to split, got {n}")
    per_level: Dict[int, list] = {int(x): [] for x in np.unique(levels)}
    for _ in range(n_splits):
        order = rng.permutation(n)
        half = n // 2
        a_mean, a_w = aggregate_subjects([subjects[i] for i in order[:half]])
        b_mean, b_w = aggregate_subjects([subjects[i] for i in order[half:]])
        both = (a_w > 0) & (b_w > 0)
        for level in per_level:
            mask = both & (levels == level)
            if mask.sum() < 3:
                continue
            rho = spearmanr(a_mean[mask], b_mean[mask])[0]
            if np.isfinite(rho):
                per_level[level].append((rho, int(mask.sum())))
    rows = []
    for level, vals in per_level.items():
        if not vals:
            continue
        rhos = np.array([v[0] for v in vals], dtype=float)
        half_r = float(rhos.mean())
        rows.append({
            "level": level,
            "level_name": _level_name(level),
            "n_splits_used": len(vals),
            "mean_pairs_per_split": float(np.mean([v[1] for v in vals])),
            "ceiling_half": half_r,
            "ceiling_half_sd": float(rhos.std()),
            # Spearman-Brown. Negative half-split correlations make the projection meaningless, so
            # they are passed through rather than "corrected" into a confident-looking number.
            "ceiling_full": float(2 * half_r / (1 + half_r)) if half_r > -1 and half_r > 0 else half_r,
        })
    return pd.DataFrame(rows).sort_values("level").reset_index(drop=True)


def diagnose(coords: np.ndarray, subjects: Sequence, manifest_images: Sequence[str],
             n_splits: int = DEFAULT_N_SPLITS,
             rng: Optional[np.random.Generator] = None) -> Dict[str, pd.DataFrame]:
    """All four tables for one GT. Keys match the CSV names the CLI writes."""
    if len(manifest_images) != coords.shape[0]:
        raise ValueError(f"manifest has {len(manifest_images)} images but the GT has "
                         f"{coords.shape[0]}; hierarchy levels would be misaligned")
    levels = hierarchy_levels(manifest_images)
    ceiling = raw_noise_ceiling(subjects, levels, n_splits=n_splits, rng=rng)
    correlation = gt_vs_raw(coords, subjects, levels)
    # The comparison the reader actually wants: how much of the achievable agreement was reached.
    merged = correlation.merge(ceiling[["level", "ceiling_full"]], on="level", how="left")
    merged["frac_of_ceiling"] = merged["spearman"] / merged["ceiling_full"]
    return {
        "level_coverage": level_coverage(subjects, levels),
        "gt_gradient": gt_gradient(coords, levels),
        "gt_vs_raw": merged,
        "noise_ceiling": ceiling,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, type=Path, help="the .npy ground-truth coordinates")
    p.add_argument("--manifest", required=True, type=Path, help="stimuli_manifest.json")
    p.add_argument("--pilot-dir", default="data", help="directory of pilot session CSVs")
    p.add_argument("--variants", default="pre",
                   help="comma-separated SHINE variants, or 'all'. Must match the GT's own subject "
                        "set: a GT is a geometry over a stimulus set, and the post-SHINE half judged "
                        "different images.")
    p.add_argument("--out", type=Path, default=None, help="directory to write the CSVs into")
    p.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from SpAM_Simulations.empirical.pilot import load_pilot_subjects

    variants = None if args.variants == "all" else tuple(args.variants.split(","))
    subjects = load_pilot_subjects(args.pilot_dir, str(args.manifest), variants=variants)
    images = json.loads(args.manifest.read_text())["images"]
    coords = np.load(args.gt)
    print(f"GT {coords.shape} from {args.gt.name}; {len(subjects)} subjects "
          f"(variants={args.variants})")

    tables = diagnose(coords, subjects, images, n_splits=args.n_splits,
                      rng=np.random.default_rng(args.seed))
    for name, frame in tables.items():
        print(f"\n--- {name} ---")
        print(frame.to_string(index=False))
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.out / f"{name}.csv", index=False)
    if args.out is not None:
        print(f"\nwrote {len(tables)} tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
