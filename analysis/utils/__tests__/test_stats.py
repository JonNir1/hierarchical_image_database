"""Tests for analysis.utils.stats: masked rank correlation, independent-
samples (Mann-Whitney U) and paired-samples (Wilcoxon) effect sizes, the image-relabeling
permutation test, and the one-line significance-summary printer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from scipy.spatial.distance import squareform

from analysis.utils.stats import (
    jackknife_bias, jackknife_ci, jackknife_se, kendall_tau_a, label_shuffle_test, masked_rank_correlation,
    mwu_effect_size, mwu_family, pearson_r, print_comparison_line, spearman_rho, wilcoxon_effect_size, z_ranks,
)

# ---------------------------------------------------------------------------
# masked_rank_correlation
# ---------------------------------------------------------------------------

_RDM_A = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
_RDM_B = np.array([1.0, np.nan, 3.0, 4.0, 6.0])


@pytest.mark.parametrize("min_overlap, underpowered", [(2, False), (5, True)])
def test_masked_rank_correlation_masks_to_jointly_covered_pairs_and_flags_underpowered(min_overlap, underpowered):
    result = masked_rank_correlation(_RDM_A, _RDM_B, min_overlap=min_overlap)

    mask = np.array([True, False, False, True, True])
    assert result["n_overlap"] == 3
    assert bool(result["underpowered"]) == underpowered
    if underpowered:
        assert np.isnan(result["kendall"])
        assert np.isnan(result["spearman"])
        return
    assert result["kendall"] == pytest.approx(kendalltau(_RDM_A[mask], _RDM_B[mask]).statistic)
    assert result["spearman"] == pytest.approx(spearmanr(_RDM_A[mask], _RDM_B[mask]).statistic)


def _brute_force_tau_a(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    s = sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i in range(n) for j in range(i + 1, n))
    return s / (n * (n - 1) / 2)


def test_kendall_tau_a_matches_brute_force_with_ties_and_the_flag_only_adds_a_key():
    rng = np.random.default_rng(0)
    a = rng.integers(1, 5, size=40).astype(float)
    b = rng.normal(size=40) + a * 0.5
    assert kendall_tau_a(a, b) == pytest.approx(_brute_force_tau_a(a, b))

    rdm_a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 2.0])
    rdm_b = np.array([1.0, np.nan, 3.0, 4.0, 6.0, 1.0])
    plain, with_a = masked_rank_correlation(rdm_a, rdm_b), masked_rank_correlation(rdm_a, rdm_b, tau_a=True)
    assert "kendall_a" not in plain and {k: with_a[k] for k in plain} == plain
    mask = ~(np.isnan(rdm_a) | np.isnan(rdm_b))
    assert with_a["kendall_a"] == pytest.approx(_brute_force_tau_a(rdm_a[mask], rdm_b[mask]))


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
    assert mwu_effect_size(b, a)["r"] == pytest.approx(-1.0)
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

@pytest.mark.parametrize("result, kwargs, expected_fragments", [
    ({"comparison": "A vs B", "U": 5.0, "p": 0.01, "r": 0.5}, {}, ["A vs B", "p=0.0100", "significant"]),
    ({"comparison": "A vs B", "U": 5.0, "p_corrected": 0.2, "r": 0.5}, {"corrected": True},
     ["p_corrected=0.2000", "not significant"]),
])
def test_print_comparison_line_reports_p_or_corrected_p(capsys, result, kwargs, expected_fragments):
    print_comparison_line(result, **kwargs)
    out = capsys.readouterr().out
    for fragment in expected_fragments:
        assert fragment in out


# ---------------------------------------------------------------------------
# label_shuffle_test
# ---------------------------------------------------------------------------

_N_IMAGES = 16


def _grouped_reference() -> np.ndarray:
    group = np.repeat(np.arange(4), 4)
    within = group[:, None] == group[None, :]
    return squareform(np.where(within, 1.0, 3.0) * (1 - np.eye(_N_IMAGES)), checks=False)


def test_label_shuffle_test_detects_planted_structure_and_is_centred_for_unrelated_incomplete_rdms():
    reference = _grouped_reference()
    rng = np.random.default_rng(0)
    planted = reference + rng.normal(scale=0.3, size=reference.size)
    planted[rng.random(planted.size) < 0.3] = np.nan
    result = label_shuffle_test(planted, reference, n_perm=200, seed=1)
    assert result.observed["kendall_a"] > 0.3 and result.p_value["kendall_a"] < 0.05

    unrelated = rng.normal(size=reference.size)
    unrelated[rng.random(unrelated.size) < 0.3] = np.nan
    null = label_shuffle_test(unrelated, reference, n_perm=200, seed=3)
    assert abs(null.null["kendall_a"].mean()) < 0.05 and null.p_value["kendall_a"] > 0.05

    complete = label_shuffle_test(reference, reference, n_perm=99)
    assert complete.observed["spearman"] == pytest.approx(1.0) and complete.p_value["spearman"] == pytest.approx(1 / 100)


def test_label_shuffle_test_statistic_subset_matches_the_full_run_and_reuses_its_cache(tmp_path):
    reference = _grouped_reference()
    perceptual = reference + np.random.default_rng(2).normal(scale=0.5, size=reference.size)
    full = label_shuffle_test(perceptual, reference, n_perm=30, seed=4)
    only = label_shuffle_test(perceptual, reference, statistics=("spearman",), n_perm=30, seed=4)
    assert list(only.observed) == ["spearman"]
    np.testing.assert_allclose(only.null["spearman"], full.null["spearman"])

    path = tmp_path / "null.npz"
    first = label_shuffle_test(perceptual, reference, n_perm=30, seed=4, cache_path=path)
    again = label_shuffle_test(perceptual, reference, n_perm=30, seed=4, cache_path=path)
    np.testing.assert_array_equal(first.null["kendall_a"], again.null["kendall_a"])


# ---------------------------------------------------------------------------
# pearson_r / spearman_rho / z_ranks
# ---------------------------------------------------------------------------

def test_pearson_and_spearman_match_scipy_for_vectors_with_ties_and_row_wise():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, size=(5, 40)).astype(float)  # integer values: many ties
    b = a + rng.normal(size=a.shape)
    assert pearson_r(a[0], b[0]) == pytest.approx(pearsonr(a[0], b[0]).statistic)
    assert spearman_rho(a[0], b[0]) == pytest.approx(spearmanr(a[0], b[0]).statistic)
    assert spearman_rho(a, b, axis=1) == pytest.approx([spearmanr(x, y).statistic for x, y in zip(a, b)])
    assert float(np.mean(z_ranks(a[0]) * z_ranks(b[0]))) == pytest.approx(spearman_rho(a[0], b[0]))


def test_correlations_of_a_constant_vector_are_nan_not_an_error():
    flat, ramp = np.ones(6), np.arange(6.0)
    assert np.isnan(pearson_r(flat, ramp)) and np.isnan(spearman_rho(ramp, flat))


def test_jackknife_of_the_mean_recovers_its_textbook_standard_error():
    x = np.random.default_rng(0).normal(size=40)
    leave_one_out_means = (x.sum() - x) / (len(x) - 1)
    assert jackknife_se(leave_one_out_means) == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)))
    assert jackknife_bias(x.mean(), leave_one_out_means) == pytest.approx(0.0, abs=1e-12)  # the mean is unbiased
    low, high = jackknife_ci(x.mean(), leave_one_out_means)
    assert (low + high) / 2 == pytest.approx(x.mean())  # centred on the estimate
