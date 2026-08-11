"""Recovery of the ground truth's closest pairs, the decision-relevant simulation-only question.

``pipeline.compute_topk_similar_pair_stability`` compares two *cohorts* to each other, which is a
reproducibility question: would a second run flag the same pairs? This module asks the other one:
does a recovered configuration find the pairs that are genuinely closest in the space the data were
generated from? That needs a ground truth, so it is answerable only in simulation.

It matters because the downstream use is stimulus construction: images that people confuse must not
land in the same stimulus. A miss (two confusable images together) damages the experiment, while a
false alarm (needlessly excluding a fine pair) is nearly free when there are 262,450 pairs to choose
from. So the quantity of interest is whether the truly-close pairs are recovered, not whether the
overall geometry is right.

**Two flavours of d-prime, deliberately.** At a matched top-``frac`` cut the recovered and
ground-truth sets are the same size, so recall, precision and the thresholded d-prime are
deterministic functions of one another and carry a single number between them. That number depends
on where the cut is placed, which is an arbitrary choice. :func:`separation_dprime` and
:func:`auc_near_pairs` are threshold-free: they ask how well the recovered distances separate the
GT-near pairs from the rest, over the whole distribution.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import norm, rankdata

from SpAM_Simulations.measures.metrics import topk_mask


def recall_at_frac(recovered: np.ndarray, gt: np.ndarray, frac: float) -> float:
    """Fraction of the GT's closest-``frac`` pairs that the recovered configuration also flags.

    With matched set sizes this equals precision and the overlap coefficient, so it is reported
    alone rather than alongside them.
    """
    g = topk_mask(gt, frac)
    r = topk_mask(recovered, frac)
    denom = int(g.sum())
    return float(np.count_nonzero(g & r) / denom) if denom else np.nan


def dprime_at_frac(recovered: np.ndarray, gt: np.ndarray, frac: float,
                   correction: str = "loglinear") -> float:
    """Signal-detection d-prime for "is this pair among the closest ``frac``".

    Hits are GT-close pairs the recovery also calls close; false alarms are GT-far pairs it calls
    close. ``correction="loglinear"`` adds 0.5 to each cell before computing rates, which keeps the
    statistic finite at ceiling and floor (an uncorrected perfect recovery gives infinite d-prime).
    """
    g = topk_mask(gt, frac)
    r = topk_mask(recovered, frac)
    hits = float(np.count_nonzero(g & r))
    n_pos = float(np.count_nonzero(g))
    false_alarms = float(np.count_nonzero(~g & r))
    n_neg = float(np.count_nonzero(~g))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    if correction == "loglinear":
        h = (hits + 0.5) / (n_pos + 1.0)
        f = (false_alarms + 0.5) / (n_neg + 1.0)
    elif correction == "none":
        h, f = hits / n_pos, false_alarms / n_neg
    else:
        raise ValueError(f"correction must be 'loglinear' or 'none', got {correction!r}")
    return float(norm.ppf(h) - norm.ppf(f))


def separation_dprime(recovered: np.ndarray, gt: np.ndarray, frac: float) -> float:
    """Standardised separation between the recovered distances of GT-near and GT-far pairs.

    ``(mean_far - mean_near) / sqrt(pooled variance)``, so positive means the truly-close pairs do
    sit closer in the recovery. Threshold-free: unlike :func:`dprime_at_frac` it uses the whole
    recovered distribution rather than only which side of a cut each pair falls on.
    """
    g = topk_mask(gt, frac)
    near, far = recovered[g], recovered[~g]
    if near.size < 2 or far.size < 2:
        return np.nan
    pooled = np.sqrt(0.5 * (near.var(ddof=1) + far.var(ddof=1)))
    return float((far.mean() - near.mean()) / pooled) if pooled > 0 else np.nan


def auc_near_pairs(recovered: np.ndarray, gt: np.ndarray, frac: float) -> float:
    """P(a random GT-near pair is recovered as closer than a random GT-far pair). Chance = 0.5.

    The rank-based (Mann-Whitney) counterpart of :func:`separation_dprime`, so it is insensitive to
    any monotone rescaling of the recovered distances - which matters because MDS output has no
    natural scale.
    """
    g = topk_mask(gt, frac)
    near, far = recovered[g], recovered[~g]
    n1, n2 = near.size, far.size
    if n1 == 0 or n2 == 0:
        return np.nan
    ranks = rankdata(np.concatenate([near, far]))
    # U counts (near, far) pairs where near ranks HIGHER, i.e. is recovered as further apart.
    u_near_greater = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(1.0 - u_near_greater / (n1 * n2))


def recovery_summary(recovered: np.ndarray, gt: np.ndarray,
                     fracs: Sequence[float] = (0.01, 0.05, 0.10)) -> List[Dict[str, float]]:
    """All four statistics at each fraction, one dict per fraction."""
    recovered = np.asarray(recovered, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if recovered.shape != gt.shape:
        raise ValueError(f"shape mismatch: recovered {recovered.shape} vs gt {gt.shape}")
    return [{
        "top_frac": float(f),
        "recall": recall_at_frac(recovered, gt, f),
        "dprime": dprime_at_frac(recovered, gt, f),
        "separation_dprime": separation_dprime(recovered, gt, f),
        "auc": auc_near_pairs(recovered, gt, f),
    } for f in fracs]
