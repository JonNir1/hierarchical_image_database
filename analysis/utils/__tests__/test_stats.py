"""Tests for analysis.utils.stats: masked rank correlation, independent-
samples (Mann-Whitney U) and paired-samples (Wilcoxon) effect sizes, and the
one-line significance-summary printer.

The masked_rank_correlation tests below were migrated from
analysis/behavioral/__tests__/test_noise_ceiling.py's former
test_kendall_spearman_* tests (that module's private _kendall_spearman() now
just calls this shared function) -- kept here rather than duplicated in both
places.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, spearmanr
from statsmodels.stats.multitest import multipletests

from analysis.utils.stats import (
    masked_rank_correlation, mwu_effect_size, mwu_family, print_comparison_line, wilcoxon_effect_size,
)

# ---------------------------------------------------------------------------
# masked_rank_correlation
# ---------------------------------------------------------------------------

def test_masked_rank_correlation_masks_to_jointly_covered_pairs():
    rdm_a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    rdm_b = np.array([1.0, np.nan, 3.0, 4.0, 6.0])
    result = masked_rank_correlation(rdm_a, rdm_b, min_overlap=2)

    mask = np.array([True, False, False, True, True])
    assert result["n_overlap"] == 3
    assert not result["underpowered"]
    expected_tau = kendalltau(rdm_a[mask], rdm_b[mask]).statistic
    expected_rho = spearmanr(rdm_a[mask], rdm_b[mask]).statistic
    assert result["kendall"] == pytest.approx(expected_tau)
    assert result["spearman"] == pytest.approx(expected_rho)


def test_masked_rank_correlation_flags_underpowered_folds():
    rdm_a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    rdm_b = np.array([1.0, np.nan, 3.0, 4.0, 6.0])
    result = masked_rank_correlation(rdm_a, rdm_b, min_overlap=5)  # only 3 jointly covered

    assert result["n_overlap"] == 3
    assert result["underpowered"]
    assert np.isnan(result["kendall"])
    assert np.isnan(result["spearman"])


def test_masked_rank_correlation_no_min_overlap_omits_underpowered_key():
    rdm_a = np.array([1.0, 2.0, np.nan, 4.0])
    rdm_b = np.array([1.0, np.nan, 3.0, 5.0])
    result = masked_rank_correlation(rdm_a, rdm_b)
    assert "underpowered" not in result
    assert result["n_overlap"] == 2


# ---------------------------------------------------------------------------
# mwu_effect_size
# ---------------------------------------------------------------------------

def test_mwu_effect_size_identical_groups_is_not_significant():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = mwu_effect_size(values, values, label_a="X", label_b="Y")
    assert result["comparison"] == "X vs Y"
    assert result["n (A)"] == 5
    assert result["p"] == pytest.approx(1.0, abs=0.05)


def test_mwu_effect_size_fully_separated_groups():
    # n=3 vs n=3: the smallest possible exact two-sided p-value is 0.1, not < 0.05
    a = [1.0, 2.0, 3.0]
    b = [10.0, 11.0, 12.0]
    result = mwu_effect_size(a, b)
    assert result["U"] == 0.0
    assert result["r"] == pytest.approx(1.0)  # maximal rank-biserial effect
    assert result["p"] == pytest.approx(0.1, abs=0.01)


def test_mwu_effect_size_drops_nan():
    a = [1.0, 2.0, np.nan, 4.0]
    b = [5.0, 6.0, 7.0]
    result = mwu_effect_size(a, b)
    assert result["n (A)"] == 3


# ---------------------------------------------------------------------------
# mwu_family
# ---------------------------------------------------------------------------

def test_mwu_family_bonferroni_matches_statsmodels_directly():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "value": np.concatenate([rng.normal(0, 1, 10), rng.normal(2, 1, 10), rng.normal(4, 1, 10)]),
        "group": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
    })
    result = mwu_family(df, "value", "group", correction="bonferroni")

    assert set(result.index) == {"a vs b", "a vs c", "b vs c"}
    raw_p = result["p"].to_numpy()
    _, expected_corrected, _, _ = multipletests(raw_p, alpha=0.05, method="bonferroni")
    np.testing.assert_allclose(result["p_corrected"].to_numpy(), expected_corrected)


def test_mwu_family_label_fn_formats_comparison_labels():
    df = pd.DataFrame({"value": [1.0, 2.0, 10.0, 11.0], "version": [1.0, 1.0, 2.0, 2.0]})
    result = mwu_family(df, "value", "version", label_fn=lambda v: f"v{v:g}")
    assert list(result.index) == ["v1 vs v2"]


# ---------------------------------------------------------------------------
# wilcoxon_effect_size
# ---------------------------------------------------------------------------

def test_wilcoxon_effect_size_all_positive_differences():
    x = np.array([2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, 1.0, 1.0])
    result = wilcoxon_effect_size(x, y)
    assert result["n"] == 4
    assert result["effect_size"] == pytest.approx(1.0)


def test_wilcoxon_effect_size_no_differences_returns_nan_without_error():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    result = wilcoxon_effect_size(x, y)
    assert result["n"] == 0
    assert np.isnan(result["statistic"])
    assert np.isnan(result["p_value"])
    assert np.isnan(result["effect_size"])


# ---------------------------------------------------------------------------
# print_comparison_line
# ---------------------------------------------------------------------------

def test_print_comparison_line_uses_p_by_default(capsys):
    result = {"comparison": "A vs B", "U": 5.0, "p": 0.01, "r": 0.5}
    print_comparison_line(result)
    out = capsys.readouterr().out
    assert "A vs B" in out
    assert "p=0.0100" in out
    assert "significant" in out


def test_print_comparison_line_uses_p_corrected_when_requested(capsys):
    result = {"comparison": "A vs B", "U": 5.0, "p_corrected": 0.2, "r": 0.5}
    print_comparison_line(result, corrected=True)
    out = capsys.readouterr().out
    assert "p_corrected=0.2000" in out
    assert "not significant" in out
