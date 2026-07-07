import json

import pandas as pd

from analysis.utils.spearman_r import distance_dict, pair_spearman_r


def _row(pairwise: list[dict] | None) -> pd.Series:
    return pd.Series({"pairwise_distances": json.dumps(pairwise) if pairwise is not None else ""})


def test_distance_dict_parses_pairwise_json():
    row = _row([
        {"src1": "a", "src2": "b", "distance": 0.2},
        {"src1": "b", "src2": "a", "distance": 0.5},  # order of src1/src2 shouldn't matter
    ])

    d = distance_dict(row)

    assert d == {frozenset({"a", "b"}): 0.5}


def test_distance_dict_handles_missing_or_empty():
    assert distance_dict(_row(None)) == {}
    assert distance_dict(_row([])) == {}
    assert distance_dict(pd.Series({})) == {}


def test_pair_spearman_r_perfect_positive():
    orig = _row([
        {"src1": "a", "src2": "b", "distance": 0.2},
        {"src1": "a", "src2": "c", "distance": 0.5},
        {"src1": "b", "src2": "c", "distance": 0.8},
    ])
    repeat = _row([
        {"src1": "a", "src2": "b", "distance": 0.1},
        {"src1": "a", "src2": "c", "distance": 0.4},
        {"src1": "b", "src2": "c", "distance": 0.9},
    ])

    assert pair_spearman_r(orig, repeat) == 1.0


def test_pair_spearman_r_perfect_negative():
    orig = _row([
        {"src1": "a", "src2": "b", "distance": 0.1},
        {"src1": "a", "src2": "c", "distance": 0.5},
        {"src1": "b", "src2": "c", "distance": 0.9},
    ])
    repeat = _row([
        {"src1": "a", "src2": "b", "distance": 0.9},
        {"src1": "a", "src2": "c", "distance": 0.5},
        {"src1": "b", "src2": "c", "distance": 0.1},
    ])

    assert pair_spearman_r(orig, repeat) == -1.0


def test_pair_spearman_r_insufficient_shared_pairs_returns_none():
    orig = _row([{"src1": "a", "src2": "b", "distance": 0.2}])
    repeat = _row([{"src1": "c", "src2": "d", "distance": 0.4}])

    assert pair_spearman_r(orig, repeat) is None
