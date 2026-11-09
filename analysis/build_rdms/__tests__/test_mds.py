"""Tests for analysis.build_rdms.behavior.mds: classical eigenspectrum, the cached weighted-
SMACOF sweep, and cross-embedding comparison. run_mds_sweep's cache-hit path
needs no R; its cache-miss (real fit) path is skipped automatically when
R/rpy2/smacof isn't available, mirroring
SpAM_Simulations/__tests__/test_pipeline_mds.py's own skip pattern."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from analysis.build_rdms.behavior.mds import (
    classical_mds_eigenspectrum, classify_mds_agreement, compare_mds_configs,
    effective_rank, run_mds_sweep,
)

# rpy2's R initialisation can fail with errors other than ImportError when R is misconfigured.
try:
    from SpAM_Simulations.core.multi_dimensional_scaling import run_mds  # noqa: F401
    _MDS_AVAILABLE = True
    _MDS_SKIP_REASON = ""
except (ImportError, IndexError, RuntimeError, OSError, AttributeError) as e:
    _MDS_AVAILABLE = False
    _MDS_SKIP_REASON = f"R/rpy2/smacof unavailable: {type(e).__name__}: {e}"

_requires_r = pytest.mark.skipif(not _MDS_AVAILABLE, reason=_MDS_SKIP_REASON)


# ---------------------------------------------------------------------------
# classical_mds_eigenspectrum / effective_rank
# ---------------------------------------------------------------------------

def test_classical_mds_eigenspectrum_fully_observed_low_rank():
    rng = np.random.default_rng(0)
    pts = rng.random((10, 2))  # true rank 2
    condensed = pdist(pts)
    eig = classical_mds_eigenspectrum(condensed)
    assert eig.shape == (10,)
    assert np.all(np.diff(eig) <= 1e-8)  # descending
    assert effective_rank(eig) <= 2  # a rank-2 embedding needs at most 2 dims


def test_classical_mds_eigenspectrum_imputes_missing_pairs():
    rng = np.random.default_rng(1)
    n_points = 8
    condensed = pdist(rng.random((n_points, 2)))
    with_missing = condensed.copy()
    with_missing[[0, 3, 5]] = np.nan
    eig = classical_mds_eigenspectrum(with_missing)
    assert eig.shape == (n_points,)
    assert not np.any(np.isnan(eig))


def test_effective_rank_threshold():
    eig = np.array([10.0, 5.0, 1e-10, -1e-9])
    assert effective_rank(eig) == 2


# ---------------------------------------------------------------------------
# classify_mds_agreement / compare_mds_configs
# ---------------------------------------------------------------------------

def test_classify_mds_agreement_equivalent():
    assert classify_mds_agreement(1e-4, 0.9999) == "EQUIVALENT"


def test_classify_mds_agreement_different_by_disparity():
    assert classify_mds_agreement(0.1, 0.999) == "DIFFERENT"


def test_classify_mds_agreement_different_by_corr():
    assert classify_mds_agreement(1e-4, 0.5) == "DIFFERENT"


def test_classify_mds_agreement_ambiguous():
    assert classify_mds_agreement(0.01, 0.97) == "ambiguous"


def test_compare_mds_configs_identical():
    rng = np.random.default_rng(2)
    conf = rng.random((10, 3))
    out = {"conf": conf, "confdist": pdist(conf)}
    result = compare_mds_configs(out, out, ndim=3)
    assert result["disparity"] == pytest.approx(0.0, abs=1e-8)
    assert result["confdist_r"] == pytest.approx(1.0, abs=1e-8)
    assert result["verdict"] == "EQUIVALENT"


def test_compare_mds_configs_differing():
    rng = np.random.default_rng(3)
    conf_a = rng.random((10, 3))
    conf_b = rng.random((10, 3))
    out_a = {"conf": conf_a, "confdist": pdist(conf_a)}
    out_b = {"conf": conf_b, "confdist": pdist(conf_b)}
    result = compare_mds_configs(out_a, out_b, ndim=3)
    assert result["disparity"] > 0
    assert result["verdict"] in ("DIFFERENT", "ambiguous")


# ---------------------------------------------------------------------------
# run_mds_sweep -- cache-hit path, needs no R
# ---------------------------------------------------------------------------

def test_run_mds_sweep_cache_hit_skips_fitting(tmp_path):
    conf = np.random.default_rng(4).random((5, 2))
    confdist = pdist(conf)
    cache_dir = tmp_path / "mds_cache"
    cache_dir.mkdir()
    np.savez(
        cache_dir / "mds_testkey_d2.npz", conf=conf, confdist=confdist,
        stress=np.array(0.25), niter=np.array(42.0), needs_more_iters=np.array(False),
    )
    # an all-NaN condensed RDM would make a real fit fail (disconnected graph) --
    # if the cache weren't hit, this test would error rather than silently pass
    condensed = np.full(len(confdist), np.nan)

    results = run_mds_sweep(condensed, [2], "testkey", cache_dir)

    assert set(results) == {2}
    np.testing.assert_array_equal(results[2]["conf"], conf)
    np.testing.assert_array_equal(results[2]["confdist"], confdist)
    assert results[2]["stress"] == pytest.approx(0.25)
    assert results[2]["niter"] == pytest.approx(42.0)
    assert results[2]["needs_more_iters"] is False


def test_run_mds_sweep_cache_filename_contract(tmp_path):
    """Cache filenames follow mds_{cache_key}_d{dim}.npz."""
    cache_dir = tmp_path / "mds_cache"
    cache_dir.mkdir()
    np.savez(
        cache_dir / "mds_pre_d3.npz",
        conf=np.zeros((4, 3)), confdist=np.zeros(6), stress=np.array(0.1),
        niter=np.array(10.0), needs_more_iters=np.array(False),
    )
    results = run_mds_sweep(np.zeros(6), [3], "pre", cache_dir)
    assert 3 in results


# ---------------------------------------------------------------------------
# run_mds_sweep -- cache-miss path, needs a real R fit
# ---------------------------------------------------------------------------

@_requires_r
def test_run_mds_sweep_fits_and_caches_on_miss(tmp_path):
    condensed = pdist(np.random.default_rng(5).random((8, 2)))
    cache_dir = tmp_path / "mds_cache"

    results = run_mds_sweep(condensed, [2], "realfit", cache_dir, max_iters=100)

    assert (cache_dir / "mds_realfit_d2.npz").exists()
    assert results[2]["conf"].shape == (8, 2)
    assert results[2]["confdist"].shape == condensed.shape
    assert results[2]["stress"] >= 0


@_requires_r
def test_run_mds_sweep_second_call_uses_cache(tmp_path):
    condensed = pdist(np.random.default_rng(6).random((8, 2)))
    cache_dir = tmp_path / "mds_cache"

    first = run_mds_sweep(condensed, [2], "cachehit", cache_dir, max_iters=100)
    cached_path = cache_dir / "mds_cachehit_d2.npz"
    mtime_before = cached_path.stat().st_mtime

    second = run_mds_sweep(condensed, [2], "cachehit", cache_dir, max_iters=100)
    assert cached_path.stat().st_mtime == mtime_before  # not rewritten
    np.testing.assert_array_equal(first[2]["conf"], second[2]["conf"])
