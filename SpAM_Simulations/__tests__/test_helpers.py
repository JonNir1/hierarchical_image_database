"""Unit tests for distance-format helpers and the condensed-index mapping."""
import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from SpAM_Simulations.core.helpers import (
    condensed_rdm_to_weighted_dists, convert_to_condensed, double_centered_gram, procrustes_m2,
)
from SpAM_Simulations.models.experiment import _condensed_pair_indices


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


@pytest.mark.parametrize("bad", [
    pytest.param(np.zeros(4), id="length-4-is-not-triangular"),
    pytest.param(np.array([[0.0, np.nan], [1.0, 0.0]]), id="nan-on-one-side-only"),
    pytest.param(np.array([[0.0, 1.0], [2.0, 0.0]]), id="finite-asymmetric"),
])
def test_invalid_input_to_convert_to_condensed_raises(bad):
    with pytest.raises(ValueError):
        convert_to_condensed(bad)


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


@pytest.mark.parametrize("condensed,expected_dists,expected_weights", [
    pytest.param([1.0, np.nan, 3.0, np.nan, 5.0], [1.0, 0.0, 3.0, 0.0, 5.0],
                 [1.0, 0.0, 1.0, 0.0, 1.0], id="nan-replaced-and-weight-zero"),
    pytest.param([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], id="fully-observed"),
])
def test_condensed_rdm_to_weighted_dists(condensed, expected_dists, expected_weights):
    dists, weights = condensed_rdm_to_weighted_dists(np.array(condensed))
    np.testing.assert_array_equal(dists, expected_dists)
    np.testing.assert_array_equal(weights, expected_weights)


def test_procrustes_m2_identical_configs_is_zero():
    rng = np.random.default_rng(0)
    conf = rng.random((10, 3))
    assert procrustes_m2(conf, conf) == pytest.approx(0.0, abs=1e-10)


def test_procrustes_m2_invariant_to_rotation_translation_scale():
    rng = np.random.default_rng(1)
    conf_a = rng.random((10, 3))
    theta = np.pi / 5
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
    conf_b = (conf_a @ rotation.T) * 2.5 + np.array([10.0, -3.0, 1.0])
    assert procrustes_m2(conf_a, conf_b) == pytest.approx(0.0, abs=1e-8)


def test_procrustes_m2_differing_configs_is_positive():
    rng = np.random.default_rng(2)
    conf_a = rng.random((10, 3))
    conf_b = rng.random((10, 3))
    assert procrustes_m2(conf_a, conf_b) > 0.0


def test_double_centered_gram_shape_and_symmetry():
    rng = np.random.default_rng(3)
    N = 6
    condensed = pdist(rng.random((N, 4)))
    gram = double_centered_gram(condensed)
    assert gram.shape == (N, N)
    np.testing.assert_allclose(gram, gram.T)


def test_double_centered_gram_matches_manual_double_centering():
    rng = np.random.default_rng(4)
    N = 5
    pts = rng.random((N, 2))
    condensed = pdist(pts)
    gram = double_centered_gram(condensed)

    sq = squareform(condensed) ** 2
    centring = np.eye(N) - np.ones((N, N)) / N
    expected = -0.5 * centring @ sq @ centring
    np.testing.assert_allclose(gram, expected)

    # the Gram matrix of the original (already-centered) points should recover
    # the same inner products up to floating point, since pts is a valid
    # Euclidean embedding of `condensed`
    centred_pts = pts - pts.mean(axis=0)
    expected_inner = centred_pts @ centred_pts.T
    np.testing.assert_allclose(gram, expected_inner, atol=1e-8)
