"""Tests for analysis.utils.equivalence (bootstrap rank-biserial and Newcombe
risk-difference equivalence tests, plus the sensitivity bound) and
analysis.utils.stats.rank_biserial, which they build on."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.utils.equivalence import (
    equivalence_sensitivity, equivalence_sensitivity_risk_difference, equivalence_test_rank_biserial,
    equivalence_test_risk_difference, print_equivalence_line, resolvable_bound,
)
from analysis.utils.stats import mwu_effect_size, rank_biserial

# ---------------------------------------------------------------------------
# rank_biserial
# ---------------------------------------------------------------------------

def test_rank_biserial_matches_mwu_effect_size_including_ties():
    a = np.array([1.0, 2.0, 2.0, 5.0, 7.0])
    b = np.array([2.0, 3.0, 6.0, 8.0, 9.0, 9.0])
    assert rank_biserial(a, b) == pytest.approx(mwu_effect_size(a, b)["r"], abs=1e-4)


def test_rank_biserial_fully_separated_groups_is_plus_or_minus_one():
    low, high = np.array([1.0, 2.0, 3.0]), np.array([10.0, 11.0, 12.0])
    assert rank_biserial(low, high) == pytest.approx(1.0)
    assert rank_biserial(high, low) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# equivalence_test_rank_biserial
# ---------------------------------------------------------------------------

def test_equivalence_rank_biserial_identical_distributions_are_equivalent():
    rng = np.random.default_rng(1)
    result = equivalence_test_rank_biserial(rng.normal(size=100), rng.normal(size=100), bound=0.3, seed=0)
    assert result["equivalent"]
    assert result["ci_low"] < result["effect"] < result["ci_high"]


def test_equivalence_rank_biserial_shifted_distributions_are_not_equivalent():
    rng = np.random.default_rng(1)
    result = equivalence_test_rank_biserial(rng.normal(0, 1, 100), rng.normal(2, 1, 100), bound=0.3, seed=0)
    assert not result["equivalent"]
    assert result["effect"] > 0.5  # A shifted below B => positive rank-biserial


def test_equivalence_rank_biserial_is_deterministic_for_a_seed_and_drops_nan():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=50), rng.normal(size=60)
    a_nan = np.append(a, np.nan)
    first = equivalence_test_rank_biserial(a_nan, b, bound=0.3, seed=7)
    second = equivalence_test_rank_biserial(a, b, bound=0.3, seed=7)
    assert first == second
    assert first["n (A)"] == 50


def test_equivalence_rank_biserial_narrower_bound_can_flip_the_verdict():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=80), rng.normal(size=80)
    assert equivalence_test_rank_biserial(a, b, bound=0.5, seed=0)["equivalent"]
    assert not equivalence_test_rank_biserial(a, b, bound=0.02, seed=0)["equivalent"]


# ---------------------------------------------------------------------------
# equivalence_test_risk_difference
# ---------------------------------------------------------------------------

def test_risk_difference_matches_hand_computed_newcombe_interval():
    # p = 0.5 in both groups, n = 80, z = 1.645: Wilson half-width 0.09044 each,
    # so the Newcombe 90% CI is 0 -/+ sqrt(2) * 0.09044 = +/-0.1279.
    result = equivalence_test_risk_difference(40, 80, 40, 80, bound=0.2)
    assert result["effect"] == 0.0
    assert result["ci_high"] == pytest.approx(0.1279, abs=1e-3)
    assert result["ci_low"] == pytest.approx(-0.1279, abs=1e-3)
    assert result["equivalent"]


def test_risk_difference_bound_below_ci_width_is_not_equivalent():
    assert not equivalence_test_risk_difference(40, 80, 40, 80, bound=0.1)["equivalent"]


def test_risk_difference_large_difference_is_not_equivalent():
    result = equivalence_test_risk_difference(70, 100, 30, 100, bound=0.15)
    assert result["effect"] == pytest.approx(0.4)
    assert not result["equivalent"]


# ---------------------------------------------------------------------------
# equivalence_sensitivity / print_equivalence_line
# ---------------------------------------------------------------------------

def test_equivalence_sensitivity_shrinks_with_larger_samples():
    small = equivalence_sensitivity(20, 20, n_sim=100, n_boot=200, seed=0)
    large = equivalence_sensitivity(80, 80, n_sim=100, n_boot=200, seed=0)
    assert 0.0 < large < small < 1.0


def test_print_equivalence_line_reports_verdict(capsys):
    result = equivalence_test_risk_difference(40, 80, 40, 80, bound=0.2, label_a="X", label_b="Y")
    print_equivalence_line(result)
    out = capsys.readouterr().out
    assert "X vs Y" in out
    assert "(equivalent)" in out


def test_equivalence_sensitivity_risk_difference_shrinks_with_n_and_peaks_near_half():
    small = equivalence_sensitivity_risk_difference(30, 30, 0.5)
    large = equivalence_sensitivity_risk_difference(120, 120, 0.5)
    skewed = equivalence_sensitivity_risk_difference(120, 120, 0.9)
    assert large < small
    assert skewed < large
    # 90% CI half-width at p=0.5, n=77 is about 0.13, and the observed difference itself
    # varies by about +/-0.10 (80th percentile), so 80% power needs roughly 0.23
    assert 0.20 < equivalence_sensitivity_risk_difference(77, 77, 0.5) < 0.26


@pytest.mark.parametrize("sensitivity, step, expected", [
    (0.27, 0.05, 0.30), (0.30, 0.05, 0.30), (0.301, 0.05, 0.35), (0.0043, 0.01, 0.01), (0.0100, 0.01, 0.01),
])
def test_resolvable_bound_rounds_up_to_the_step(sensitivity, step, expected):
    assert resolvable_bound(sensitivity, step=step) == pytest.approx(expected)
