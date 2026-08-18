"""Tests for the Python port of the deployed screening gate.

The cases mirror ``SpAM_Task/__tests__/utils.test.js``'s ``evaluateScreening`` suite one for one,
using the same thresholds and the same inputs, because the whole value of this module is that it
decides identically to the browser. Where a case is Python-only it says so.
"""
import json

import numpy as np
import pandas as pd
import pytest

from SpAM_Simulations.empirical import screening_audit as sa

# The JS suite's config, flattened into this module's threshold dict.
CFG = {
    "min_move_item_ratio": 0.75,
    "min_pairwise_distance_sd": 0.04,
    "min_reliability": 0.30,
    "median_reliability": 0.30,
    "move_ratio_max_fail_rate": 0.30,
    "distance_sd_max_fail_rate": 0.30,
}

_COLUMNS = ["pairwise_distances", "num_moves", "reliability"]


def _pw(distances):
    """A pairwise_distances JSON string over as many images as the distances imply."""
    names = "abcdefghijklmnopqrstuvwxyz"
    rows, k = [], 0
    n = 2
    while n * (n - 1) // 2 < len(distances):
        n += 1
    for i in range(n):
        for j in range(i + 1, n):
            if k >= len(distances):
                break
            rows.append({"src1": f"./images/pre_shine/{names[i]}.png",
                         "src2": f"./images/pre_shine/{names[j]}.png",
                         "distance": distances[k]})
            k += 1
    return json.dumps(rows)


def _trials(specs):
    """Frame of main trials. Each spec is (distances, num_moves, reliability|None)."""
    if not specs:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in _COLUMNS})
    return pd.DataFrame([
        {"pairwise_distances": _pw(d), "num_moves": m,
         "reliability": np.nan if r is None else r}
        for d, m, r in specs])


# A three-image trial whose distances are well spread and whose moves clear any ratio.
GOOD = ([0.1, 0.5, 0.9], 20, None)
# Same trial, but barely moved.
LAZY = ([0.1, 0.5, 0.9], 1, None)
# Moved plenty, but everything piled up: SD below any sane threshold.
PILED = ([0.30, 0.30, 0.30001], 20, None)


class TestMirrorsTheJsSuite:
    def test_skips_reliability_criteria_when_no_repeat_has_completed(self):
        out = sa.evaluate_screening(_trials([GOOD]), CFG)
        assert out["min_reliability"] is None
        assert out["median_reliability"] is None
        assert not any("reliability" in r for r in out["reasons"])
        assert out["pass"] is True

    def test_fails_on_the_minimum_reliability_not_the_median(self):
        """One bad repeat is enough; the median of these would pass comfortably."""
        out = sa.evaluate_screening(_trials([]), CFG, reliabilities=[0.9, 0.95, 0.05])
        assert out["min_reliability"] == 0.05
        assert out["pass"] is False
        assert any("minimum reliability" in r for r in out["reasons"])

    def test_min_and_median_criteria_are_independent(self):
        only_min = sa.evaluate_screening(_trials([]), CFG, reliabilities=[0.05, 0.9, 0.95])
        assert any("minimum reliability" in r for r in only_min["reasons"])
        assert not any("median reliability" in r for r in only_min["reasons"])

        lenient = {**CFG, "min_reliability": 0.05}
        only_median = sa.evaluate_screening(_trials([]), lenient, reliabilities=[0.10, 0.15, 0.20])
        assert not any("minimum reliability" in r for r in only_median["reasons"])
        assert any("median reliability" in r for r in only_median["reasons"])

    def test_null_disables_each_criterion_individually(self):
        disabled = {**CFG, "min_reliability": None, "median_reliability": None,
                    "move_ratio_max_fail_rate": None, "distance_sd_max_fail_rate": None}
        out = sa.evaluate_screening(_trials([LAZY, PILED]), disabled, reliabilities=[0.01, 0.02])
        assert out["pass"] is True
        assert out["reasons"] == []
        # The statistics are still reported; only the verdict is disabled. LAZY fails the move
        # ratio and PILED fails the spread, one each.
        assert (out["move_ratio_fails"], out["distance_sd_fails"]) == (1, 1)
        assert out["min_reliability"] == 0.01

    def test_comparisons_are_strict_so_a_value_exactly_at_the_threshold_passes(self):
        at_threshold = sa.evaluate_screening(_trials([]), CFG, reliabilities=[0.30])
        assert at_threshold["pass"] is True


class TestFailRateAttribution:
    def test_move_ratio_and_distance_sd_are_counted_separately(self):
        """`qc_flag` collapses these two into one boolean, so the audit must recompute them."""
        out = sa.evaluate_screening(_trials([GOOD, LAZY, PILED, GOOD]), CFG)
        assert out["move_ratio_fails"] == 1
        assert out["distance_sd_fails"] == 1
        assert out["n_trials"] == 4

    def test_counts_are_reported_beside_rates(self):
        """The same 0.13 threshold means 1-of-8 in the screening block and 1-of-14 in the other,
        so a rate alone cannot be compared across blocks."""
        out = sa.evaluate_screening(_trials([LAZY] + [GOOD] * 7), CFG)
        assert out["move_ratio_fails"] == 1
        assert out["move_ratio_fail_rate"] == pytest.approx(1 / 8)

    def test_criteria_of_attributes_each_reason(self):
        out = sa.evaluate_screening(_trials([LAZY, PILED]), CFG, reliabilities=[0.01])
        criteria = sa.criteria_of(out["reasons"])
        assert set(criteria) == {"move_ratio", "distance_sd", "reliability"}

    def test_simulable_marks_only_the_reliability_criterion(self):
        """Every simulated subject touches every image, so move-ratio can never fire there."""
        assert sa.SIMULABLE["reliability"] is True
        assert sa.SIMULABLE["move_ratio"] is False
        assert sa.SIMULABLE["distance_sd"] is False


class TestThresholdsComeFromTheDeployedConfig:
    def test_load_thresholds_reads_the_task_config(self):
        thr = sa.load_thresholds()
        assert set(thr) == set(CFG)
        # The deployed gate at the time of writing; if this changes, the report must change with it.
        assert thr["min_reliability"] == 0.0
        assert thr["median_reliability"] is None

    def test_reliability_defaults_to_the_frames_own_column(self):
        trials = _trials([([0.1, 0.5, 0.9], 20, 0.02), GOOD])
        out = sa.evaluate_screening(trials, CFG)
        assert out["n_repeats_scored"] == 1
        assert out["min_reliability"] == 0.02
        assert any("minimum reliability" in r for r in out["reasons"])


def test_trial_sd_matches_a_hand_computation():
    """utils.js:computeSD uses the sample SD (n-1), and so must this."""
    assert sa.trial_sd(_pw([0.1, 0.5, 0.9])) == pytest.approx(np.std([0.1, 0.5, 0.9], ddof=1))
    # Fewer than two pairs is defined as 0, matching the JS guard rather than raising.
    assert sa.trial_sd(_pw([0.4])) == 0.0
