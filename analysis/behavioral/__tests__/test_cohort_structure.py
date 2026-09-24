"""Tests for analysis.behavioral.cohort_structure's cohort-label permutation test,
using a cheap synthetic RDM builder (mean of per-subject vectors) in place of the
real trial-aggregating builder."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.behavioral.cohort_structure import equivalence_test_cohort_agreement, permutation_test_cohort_agreement

_N_PAIRS = 300


def _subjects(n_per_cohort: int, same_structure: bool, seed: int = 0) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    base_a = rng.normal(size=_N_PAIRS)
    base_b = base_a if same_structure else rng.normal(size=_N_PAIRS)
    subject_rdms, ids_a, ids_b = {}, [], []
    for i in range(n_per_cohort):
        subject_rdms[f"a{i}"] = base_a + rng.normal(scale=2.0, size=_N_PAIRS)
        subject_rdms[f"b{i}"] = base_b + rng.normal(scale=2.0, size=_N_PAIRS)
        ids_a.append(f"a{i}")
        ids_b.append(f"b{i}")
    return subject_rdms, ids_a, ids_b


class _MeanBuilder:
    calls = 0

    def __init__(self, subject_rdms: dict[str, np.ndarray]) -> None:
        self.subject_rdms = subject_rdms

    def __call__(self, ids) -> np.ndarray:
        self.calls += 1
        return np.mean([self.subject_rdms[i] for i in ids], axis=0)


def test_exchangeable_cohorts_give_a_non_small_p():
    subject_rdms, ids_a, ids_b = _subjects(12, same_structure=True)
    result = permutation_test_cohort_agreement(_MeanBuilder(subject_rdms), ids_a, ids_b, n_perm=100, seed=1)
    assert result["spearman"]["p"] > 0.05
    assert result["null"].shape == (100, 2)


def test_cohorts_with_different_structure_give_a_small_p():
    subject_rdms, ids_a, ids_b = _subjects(12, same_structure=False)
    result = permutation_test_cohort_agreement(_MeanBuilder(subject_rdms), ids_a, ids_b, n_perm=100, seed=1)
    assert result["spearman"]["p"] == pytest.approx(1 / 101)
    assert result["spearman"]["observed"] < result["spearman"]["null_low"]


def test_cache_round_trip_and_fingerprint_invalidation(tmp_path):
    subject_rdms, ids_a, ids_b = _subjects(6, same_structure=True)
    cache = tmp_path / "null.npz"
    first = permutation_test_cohort_agreement(
        _MeanBuilder(subject_rdms), ids_a, ids_b, n_perm=20, seed=3, cache_path=cache,
    )
    assert cache.exists()

    counting = _MeanBuilder(subject_rdms)
    second = permutation_test_cohort_agreement(counting, ids_a, ids_b, n_perm=20, seed=3, cache_path=cache)
    assert counting.calls == 2  # observed cohorts only, no permutation draws
    np.testing.assert_array_equal(first["null"], second["null"])

    other_seed = permutation_test_cohort_agreement(
        _MeanBuilder(subject_rdms), ids_a, ids_b, n_perm=20, seed=4, cache_path=cache,
    )
    assert not np.array_equal(first["null"], other_seed["null"])


def test_equivalence_exchangeable_cohorts_are_equivalent_within_the_resolvable_bound():
    subject_rdms, ids_a, ids_b = _subjects(15, same_structure=True)
    builder = _MeanBuilder(subject_rdms)
    permutation = permutation_test_cohort_agreement(builder, ids_a, ids_b, n_perm=50, seed=1)
    result = equivalence_test_cohort_agreement(
        builder, ids_a, ids_b, permutation, n_groups=5, n_null=5, seed=1,
    )["spearman"]
    assert result["ci_low"] < result["effect"] < result["ci_high"]
    assert result["bound"] >= result["sensitivity"]
    assert result["equivalent"]


def test_equivalence_cohorts_with_different_structure_are_not_equivalent():
    subject_rdms, ids_a, ids_b = _subjects(15, same_structure=False)
    builder = _MeanBuilder(subject_rdms)
    permutation = permutation_test_cohort_agreement(builder, ids_a, ids_b, n_perm=50, seed=1)
    result = equivalence_test_cohort_agreement(
        builder, ids_a, ids_b, permutation, n_groups=5, n_null=5, seed=1,
    )["spearman"]
    assert result["effect"] < -0.1
    assert not result["equivalent"]
