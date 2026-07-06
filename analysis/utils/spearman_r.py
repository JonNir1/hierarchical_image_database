"""
Spearman R between a trial's pairwise image distances and its verbatim
repeat's, used to quantify test-retest reliability of a SpAM arrangement.

Usage (from repo root):
    from analysis.utils.spearman_r import pair_spearman_r

    r = pair_spearman_r(orig_row, repeat_row)  # float, or None if too few shared pairs
"""
from __future__ import annotations

import json

import pandas as pd
from scipy.stats import spearmanr


def distance_dict(row: pd.Series) -> dict[frozenset, float]:
    """Parse a trial row's pairwise_distances JSON into {frozenset({src1, src2}): distance}."""
    raw = row.get("pairwise_distances", None)
    if raw is None or pd.isna(raw) or raw == "":
        return {}
    items = json.loads(raw)
    return {frozenset((d["src1"], d["src2"])): d["distance"] for d in items}


def pair_spearman_r(orig_row: pd.Series, repeat_row: pd.Series) -> float | None:
    """Spearman R between an original trial's and its repeat's pairwise image distances."""
    d_orig, d_repeat = distance_dict(orig_row), distance_dict(repeat_row)
    keys = sorted(d_orig.keys() & d_repeat.keys(), key=lambda k: sorted(k))
    if len(keys) < 2:
        return None
    r, _p = spearmanr([d_orig[k] for k in keys], [d_repeat[k] for k in keys])
    return r
