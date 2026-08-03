"""
Tests for analysis.rdms.validate_rdms: the individual check_* invariant
functions and the run_all_checks() orchestrator.

Unlike test_rdm_outputs.py (which validates the *real* generated D_<name>.npy
files and skips when they're absent), these tests exercise validate_rdms.py's
own logic against small synthetic condensed-distance arrays, so they run the
same way regardless of whether any real RDM has been built -- including in
the lightweight CI job, which never runs build_all.
"""
from __future__ import annotations

import numpy as np
import pytest

import analysis.rdms.common as common
import analysis.rdms.validate_rdms as vr
from analysis.rdms.validate_rdms import (
    CheckFailed,
    check_clip,
    check_clip_correlation,
    check_sem_km,
    check_sem_wn,
    check_sens,
    check_sens_correlation,
    check_universal,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Small synthetic dataset constants (used to patch module-level values)
# ---------------------------------------------------------------------------
_N = 6
_LEN = _N * (_N - 1) // 2  # 15


def _valid_condensed() -> np.ndarray:
    """A well-formed condensed distance vector: positive, finite, length _LEN."""
    rng = np.random.default_rng(0)
    points = rng.random((_N, 3)) * 10
    from scipy.spatial.distance import pdist
    return pdist(points, metric="euclidean")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_n(monkeypatch):
    """Patch validate_rdms' module-level _N / _EXPECTED_LEN to the small test size."""
    monkeypatch.setattr(vr, "_N", _N)
    monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)


@pytest.fixture()
def patched_env(tmp_path, monkeypatch, patched_n):
    """
    Full environment for run_all_checks() integration tests: redirects both
    common.py's and validate_rdms.py's module-level RESULTS_DIR to a temp dir
    (they hold independent `from ... import RESULTS_DIR` bindings) and shrinks
    both modules' expected N/length so common.save_rdm() can write valid
    small synthetic RDMs that validate_rdms' checks will accept.
    """
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(common, "_EXPECTED_LEN", _LEN)
    monkeypatch.setattr(common, "_EXPECTED_N", _N)
    monkeypatch.setattr(vr, "RESULTS_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# check_universal
# ---------------------------------------------------------------------------

class TestCheckUniversal:
    def test_valid_condensed_array_passes(self, patched_n):
        check_universal("x", _valid_condensed())  # must not raise

    def test_wrong_length_raises(self, patched_n):
        with pytest.raises(CheckFailed, match="length"):
            check_universal("x", _valid_condensed()[:-1])

    def test_nan_raises(self, patched_n):
        d = _valid_condensed()
        d[0] = np.nan
        with pytest.raises(CheckFailed, match="NaN"):
            check_universal("x", d)

    def test_inf_raises(self, patched_n):
        d = _valid_condensed()
        d[0] = np.inf
        with pytest.raises(CheckFailed, match="Inf"):
            check_universal("x", d)

    def test_negative_value_raises(self, patched_n):
        d = _valid_condensed()
        d[0] = -1.0
        with pytest.raises(CheckFailed, match="negative"):
            check_universal("x", d)


# ---------------------------------------------------------------------------
# check_sens
# ---------------------------------------------------------------------------

class TestCheckSens:
    def test_positive_distances_pass(self):
        check_sens(_valid_condensed())  # must not raise

    def test_all_zero_raises(self):
        with pytest.raises(CheckFailed, match="zero"):
            check_sens(np.zeros(_LEN))


# ---------------------------------------------------------------------------
# check_sens_correlation
# ---------------------------------------------------------------------------

class TestCheckSensCorrelation:
    def test_correlated_passes(self):
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        check_sens_correlation(d, d.copy())  # rho = 1.0 > 0.5, must not raise

    def test_anticorrelated_raises(self):
        d_pre = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d_post = d_pre[::-1].copy()  # rho = -1.0 < 0.5
        with pytest.raises(CheckFailed, match="Spearman"):
            check_sens_correlation(d_pre, d_post)


# ---------------------------------------------------------------------------
# check_sem_km
# ---------------------------------------------------------------------------

class TestCheckSemKm:
    def test_valid_integer_distances_pass(self):
        d = np.array([1.0, 2.0, 3.0, 1.0, 4.0, 2.0])
        check_sem_km(d)  # must not raise

    def test_non_integer_raises(self):
        d = np.array([1.0, 2.5, 3.0])
        with pytest.raises(CheckFailed, match="integer"):
            check_sem_km(d)

    def test_min_below_one_raises(self):
        d = np.array([0.0, 2.0, 3.0])
        with pytest.raises(CheckFailed, match="< 1"):
            check_sem_km(d)

    def test_max_above_bound_raises(self):
        # default _KM_MAX_DEPTH = 8 -> bound is 2*8 = 16
        d = np.array([1.0, 2.0, 17.0])
        with pytest.raises(CheckFailed, match="max KM distance"):
            check_sem_km(d)


# ---------------------------------------------------------------------------
# check_sem_wn
# ---------------------------------------------------------------------------

class TestCheckSemWn:
    def test_valid_range_and_low_fallback_fraction_passes(self):
        # 20 values, only 1 (5%) at the fallback ceiling of 30.0
        d = np.array([5.0] * 19 + [30.0])
        check_sem_wn(d)  # must not raise

    def test_max_above_fallback_raises(self):
        d = np.array([5.0, 10.0, 31.0])
        with pytest.raises(CheckFailed, match="max WN distance"):
            check_sem_wn(d)

    def test_high_fallback_fraction_raises(self):
        # 20 values, 4 (20%) at the fallback ceiling -- above the 10% threshold
        d = np.array([5.0] * 16 + [30.0] * 4)
        with pytest.raises(CheckFailed, match="fallback"):
            check_sem_wn(d)


# ---------------------------------------------------------------------------
# check_clip
# ---------------------------------------------------------------------------

class TestCheckClip:
    def test_valid_range_passes(self):
        d = np.array([0.0, 1.0, 2.0])
        check_clip(d)  # must not raise

    def test_above_bound_raises(self):
        d = np.array([0.5, 1.0, 2.5])
        with pytest.raises(CheckFailed, match="max CLIP cosine distance"):
            check_clip(d)


# ---------------------------------------------------------------------------
# check_clip_correlation
# ---------------------------------------------------------------------------

class TestCheckClipCorrelation:
    def test_correlated_passes(self):
        d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        check_clip_correlation(d, d.copy())  # rho = 1.0 > 0.9, must not raise

    def test_anticorrelated_raises(self):
        d_pre = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        d_post = d_pre[::-1].copy()  # rho = -1.0 < 0.9
        with pytest.raises(CheckFailed, match="Spearman"):
            check_clip_correlation(d_pre, d_post)


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_absent_rdm_is_skipped_not_failed(self, patched_env):
        """A target with no saved .npy is skipped, not counted as a failure."""
        d = _valid_condensed()
        common.save_rdm("sens_pre", d, metric="euclidean", source="tests")
        common.save_rdm("sens_post", d.copy(), metric="euclidean", source="tests")
        # "sem_km" is never saved -- must be silently skipped, not fail the run.
        assert run_all_checks(["sens_pre", "sens_post", "sem_km"]) is True

    def test_load_error_surfaces_as_failure(self, patched_env):
        """A .npy written outside save_rdm() (no metadata record) fails to load."""
        np.save(patched_env / "D_sens_pre.npy", _valid_condensed())
        assert run_all_checks(["sens_pre"]) is False

    def test_failing_check_returns_false(self, patched_env):
        """A validly-saved RDM that fails its per-RDM check makes the run fail."""
        common.save_rdm("sens_pre", np.zeros(_LEN), metric="euclidean", source="tests")
        assert run_all_checks(["sens_pre"]) is False

    def test_all_present_and_passing_returns_true(self, patched_env):
        d = _valid_condensed()
        common.save_rdm("sens_pre", d, metric="euclidean", source="tests")
        common.save_rdm("sens_post", d.copy(), metric="euclidean", source="tests")
        assert run_all_checks(["sens_pre", "sens_post"]) is True
