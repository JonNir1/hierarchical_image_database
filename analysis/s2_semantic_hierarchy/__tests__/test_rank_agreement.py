"""Tests for analysis.s2_semantic_hierarchy.rank_agreement: the NaN-aware correlation matrix."""
from __future__ import annotations

import numpy as np
import pytest

import analysis.s2_semantic_hierarchy.rank_agreement as ra


class TestCorrelationMatrix:
    def test_diagonal_is_one_and_symmetric(self):
        a = np.array([1.0, 2, 3, 4, 5, 6])
        rdms = {"a": a, "b": a[::-1].copy(), "c": np.array([1.0, 1, 2, 2, 3, 3])}
        corr = ra.correlation_matrix(rdms, "spearman")
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr.to_numpy(), corr.to_numpy().T)

    def test_reversed_vector_is_perfectly_anticorrelated(self):
        a = np.array([1.0, 2, 3, 4, 5, 6])
        corr = ra.correlation_matrix({"a": a, "rev": a[::-1].copy()}, "spearman")
        assert corr.loc["a", "rev"] == pytest.approx(-1.0)

    def test_nan_entries_equal_the_complete_subset(self):
        rng = np.random.default_rng(0)
        a, b = rng.random(45), rng.random(45)
        a_nan = a.copy()
        a_nan[::5] = np.nan
        keep = ~np.isnan(a_nan)
        for stat in ("kendall_a", "spearman"):
            full = ra.correlation_matrix({"a": a_nan, "b": b}, stat).loc["a", "b"]
            subset = ra.correlation_matrix({"a": a[keep], "b": b[keep]}, stat).loc["a", "b"]
            assert full == pytest.approx(subset)
