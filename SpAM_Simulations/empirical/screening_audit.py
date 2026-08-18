"""Score the deployed task's screening criteria against any block of collected trials.

A Python port of ``SpAM_Task/js/utils.js:evaluateScreening`` plus the ``computeMainQcFlag``
components it reads. The task runs that rule **once**, on the screening block, to decide who
continues. This module runs the same rule on whichever trials it is handed, which is what makes the
interesting question answerable:

* an **early fail** failed the gate in-task and was paid the reduced rate;
* a **false positive** passed the gate, was paid in full, and then failed the *same* rule on their
  experimental block.

The second group is invisible to the deployed task. The simulation can produce its own version of it
(apply the threshold to the main-stage repeats of retained subjects), which is what makes the two
comparable - but only for the reliability criterion. See :data:`SIMULABLE`.

**Thresholds come from ``task_config.json``**, never from constants here, so an audit cannot drift
from the deployed gate. One subtlety matters: the two fail-*rate* thresholds are fractions over a
denominator that differs between blocks, 8 main trials in the screening block against 14 in the
experimental one, so ``0.13`` means "at most 1 of 8" in one and "at most 1 of 14" in the other.
Counts are reported beside rates for that reason.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.utils.parser import parse_pairwise_distances

# Mirrors utils.js:roundTo's default, so a Python-side comparison lands on the same side of a
# threshold as the browser's did.
_ROUND_DECIMALS = 10

DEFAULT_CONFIG = "SpAM_Task/task_config.json"


def load_thresholds(config_path: str = DEFAULT_CONFIG) -> Dict[str, object]:
    """The deployed gate, read from the task's own config.

    Returns the four nullable threshold values plus the two per-trial constants they are applied
    against, flattened into one dict so callers need not know the config's nesting.
    """
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    thr = cfg["screening_block"]["thresholds"]
    et = cfg["experimental_trials"]
    return {
        "move_ratio_max_fail_rate": thr["move_ratio_max_fail_rate"],
        "distance_sd_max_fail_rate": thr["distance_sd_max_fail_rate"],
        "min_reliability": thr["min_reliability"],
        "median_reliability": thr["median_reliability"],
        "min_move_item_ratio": et["min_move_item_ratio"],
        "min_pairwise_distance_sd": et["min_pairwise_distance_sd"],
    }


def _round(value: float) -> float:
    return value if not np.isfinite(value) else float(round(value, _ROUND_DECIMALS))


def trial_sd(pairwise_json: str) -> float:
    """SD of a trial's normalised pairwise distances, matching ``utils.js:computeSD`` (ddof=1)."""
    values = list(parse_pairwise_distances(pairwise_json).values())
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def trial_n_items(pairwise_json: str) -> int:
    """Images in a trial, recovered from the pairs it reports."""
    images = set()
    for a, b in parse_pairwise_distances(pairwise_json):
        images.add(a)
        images.add(b)
    return len(images)


def evaluate_screening(trials: pd.DataFrame, thresholds: Dict[str, object],
                       reliabilities: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """Apply the deployed gate to one block of one subject's main trials.

    ``trials`` is a parser-style frame already restricted to that subject and block, catch trials
    excluded. ``reliabilities`` defaults to the frame's own non-null ``reliability`` column, which
    the task writes on repeat trials only.

    Mirrors the JS in the details that decide edge cases: comparisons are strict, so a value exactly
    at the threshold passes; a null threshold disables its criterion; and the two reliability
    criteria are skipped rather than failed when no repeat has completed.
    """
    n = len(trials)
    n_items = trials["pairwise_distances"].map(trial_n_items)
    sds = trials["pairwise_distances"].map(trial_sd)
    move_fails = int((trials["num_moves"] < thresholds["min_move_item_ratio"] * n_items).sum())
    sd_fails = int((sds < thresholds["min_pairwise_distance_sd"]).sum())
    move_rate = _round(0.0 if n == 0 else move_fails / n)
    sd_rate = _round(0.0 if n == 0 else sd_fails / n)

    if reliabilities is None:
        col = trials["reliability"] if "reliability" in trials.columns else pd.Series(dtype=float)
        reliabilities = [float(v) for v in col.dropna()]
    reliabilities = [v for v in reliabilities if np.isfinite(v)]
    min_rel = _round(min(reliabilities)) if reliabilities else None
    median_rel = _round(float(np.median(reliabilities))) if reliabilities else None

    reasons: List[str] = []
    if (thresholds["move_ratio_max_fail_rate"] is not None
            and move_rate > thresholds["move_ratio_max_fail_rate"]):
        reasons.append(f"move-ratio fail rate {move_rate:.3f} exceeds move_ratio_max_fail_rate "
                       f"({thresholds['move_ratio_max_fail_rate']})")
    if (thresholds["distance_sd_max_fail_rate"] is not None
            and sd_rate > thresholds["distance_sd_max_fail_rate"]):
        reasons.append(f"distance-SD fail rate {sd_rate:.3f} exceeds distance_sd_max_fail_rate "
                       f"({thresholds['distance_sd_max_fail_rate']})")
    if (thresholds["min_reliability"] is not None and min_rel is not None
            and min_rel < thresholds["min_reliability"]):
        reasons.append(f"minimum reliability {min_rel:.3f} is below min_reliability "
                       f"({thresholds['min_reliability']})")
    if (thresholds["median_reliability"] is not None and median_rel is not None
            and median_rel < thresholds["median_reliability"]):
        reasons.append(f"median reliability {median_rel:.3f} is below median_reliability "
                       f"({thresholds['median_reliability']})")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "n_trials": n,
        "move_ratio_fail_rate": move_rate, "move_ratio_fails": move_fails,
        "distance_sd_fail_rate": sd_rate, "distance_sd_fails": sd_fails,
        "min_reliability": min_rel, "median_reliability": median_rel,
        "n_repeats_scored": len(reliabilities),
    }


# The criterion each reason string belongs to, so failures can be attributed rather than only
# counted. Keyed by the prefix `evaluate_screening` builds its messages from.
_CRITERION_OF = (
    ("move-ratio", "move_ratio"),
    ("distance-SD", "distance_sd"),
    ("minimum reliability", "reliability"),
    ("median reliability", "reliability"),
)

# Which criteria the simulation's screening model can produce at all.
#
# `task_v4_experiment` screens on per-repeat test-retest and nothing else, and its subjects are
# arrangements of every image in the trial - so a simulated subject can never fail the move-ratio
# check, and arrangement spread is not modelled. Recorded here rather than in prose so the report's
# attribution table cannot drift from the truth.
SIMULABLE = {"reliability": True, "move_ratio": False, "distance_sd": False}


def criteria_of(reasons: Sequence[str]) -> List[str]:
    """The distinct criteria a list of failure reasons implicates, in the order they appear."""
    out: List[str] = []
    for reason in reasons:
        for prefix, name in _CRITERION_OF:
            if reason.startswith(prefix) and name not in out:
                out.append(name)
    return out
