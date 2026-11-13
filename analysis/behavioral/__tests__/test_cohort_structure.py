"""Tests for analysis.behavioral.cohort_structure's cohort-label permutation test,
using a cheap synthetic RDM builder (mean of per-subject vectors) in place of the
real trial-aggregating builder."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from analysis.behavioral.cohort_structure import (
    equivalence_test_cohort_agreement, permutation_test_cohort_agreement, rdm_from_subject_trials,
)

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


class _Cohorts(NamedTuple):
    builder: _MeanBuilder
    ids_a: list[str]
    ids_b: list[str]
    permutation: dict


def _cohorts(same_structure: bool) -> _Cohorts:
    subject_rdms, ids_a, ids_b = _subjects(15, same_structure=same_structure)
    builder = _MeanBuilder(subject_rdms)
    permutation = permutation_test_cohort_agreement(builder, ids_a, ids_b, n_perm=100, seed=1)
    return _Cohorts(builder, ids_a, ids_b, permutation)


@pytest.fixture(scope="module")
def exchangeable_cohorts() -> _Cohorts:
    return _cohorts(same_structure=True)


@pytest.fixture(scope="module")
def different_cohorts() -> _Cohorts:
    return _cohorts(same_structure=False)


def test_exchangeable_cohorts_give_a_non_small_p(exchangeable_cohorts):
    result = exchangeable_cohorts.permutation
    assert result["spearman"]["p"] > 0.05
    assert result["null"].shape == (100, 2)


def test_cohorts_with_different_structure_give_a_small_p(different_cohorts):
    result = different_cohorts.permutation
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


def _equivalence(cohorts: _Cohorts) -> dict:
    return equivalence_test_cohort_agreement(
        cohorts.builder, cohorts.ids_a, cohorts.ids_b, cohorts.permutation, n_groups=5, n_null=5, seed=1,
    )["spearman"]


def test_equivalence_exchangeable_cohorts_are_equivalent_within_the_resolvable_bound(exchangeable_cohorts):
    result = _equivalence(exchangeable_cohorts)
    assert result["ci_low"] < result["effect"] < result["ci_high"]
    assert result["bound"] >= result["sensitivity"]
    assert result["equivalent"]


def test_equivalence_cohorts_with_different_structure_are_not_equivalent(different_cohorts):
    result = _equivalence(different_cohorts)
    assert result["effect"] < -0.1
    assert not result["equivalent"]


def test_rdm_from_subject_trials_pools_selected_subjects_with_simple_rescaling():
    pairs = np.array([0, 1, 2])  # condensed indices of the first 3 of 6 pairs for 4 conditions
    trials_by_subject = {
        "a": [(pairs, np.array([1.0, 2.0, 3.0]))],
        "b": [(pairs, np.array([2.0, 4.0, 6.0]))],
        "c": [(np.array([3, 4, 5]), np.array([9.0, 1.0, 1.0]))],  # not selected
    }
    rdm = rdm_from_subject_trials(["a", "b"], trials_by_subject, n_conds=4)

    assert rdm.shape == (6,)  # n(n-1)/2
    # Proportional trials rescale to the same unit-norm vector, so pooling keeps it.
    np.testing.assert_allclose(rdm[:3], np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0))
    assert np.isnan(rdm[3:]).all()
