"""Unit tests for distance-format helpers and the condensed-index mapping."""
import numpy as np
import pytest
from scipy.spatial.distance import squareform

from SpAM_Simulations.helpers import convert_to_condensed
from SpAM_Simulations.experiment import _condensed_pair_indices


def test_condensed_input_returns_equal_copy():
    v = np.arange(10, dtype=np.float32)  # length 10 -> N=5 condensed
    out = convert_to_condensed(v)
    assert np.array_equal(out, v)
    assert out is not v  # a copy, so mutation does not leak back
    out[0] = 999
    assert v[0] == 0


def test_square_input_condensed():
    rng = np.random.default_rng(0)
    N = 7
    cond = rng.random(N * (N - 1) // 2)
    sq = squareform(cond)
    np.testing.assert_array_equal(convert_to_condensed(sq), cond)


def test_invalid_condensed_length_raises():
    with pytest.raises(ValueError):
        convert_to_condensed(np.zeros(4))  # 4 is not a triangular number


def test_nan_asymmetric_square_raises():
    # a NaN present on one side of the diagonal but not the other is rejected
    m = np.array([[0.0, np.nan], [1.0, 0.0]])
    with pytest.raises(ValueError):
        convert_to_condensed(m)


def test_finite_asymmetric_square_raises():
    # a finite but non-symmetric matrix must be rejected (no NaNs to mask the asymmetry)
    m = np.array([[0.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError):
        convert_to_condensed(m)


def test_symmetric_nan_square_passes():
    # a symmetric NaN mask with matching finite entries is accepted
    m = np.array([[0.0, np.nan, 3.0],
                  [np.nan, 0.0, 5.0],
                  [3.0, 5.0, 0.0]])
    out = convert_to_condensed(m)
    np.testing.assert_array_equal(np.isnan(out), [True, False, False])
    assert out[1] == 3.0 and out[2] == 5.0


@pytest.mark.parametrize("N", [3, 7, 12, 30])
def test_condensed_pair_indices_match_squareform(N):
    # the helper must map every unordered pair to scipy's condensed position
    sq = squareform(np.arange(N * (N - 1) // 2, dtype=float))
    rng = np.random.default_rng(N)
    a = rng.integers(0, N, size=50)
    b = rng.integers(0, N, size=50)
    keep = a != b
    a, b = a[keep], b[keep]
    cond_vec = squareform(sq)
    idx = _condensed_pair_indices(a, b, N)
    np.testing.assert_array_equal(cond_vec[idx], [sq[i, j] for i, j in zip(a, b)])
