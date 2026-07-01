"""Tests for analysis.rdms.validate_rdms: the pure numpy/scipy invariant checks
and the run_all_checks orchestration (load/skip/failure aggregation)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

import analysis.rdms.common as common
import analysis.rdms.validate_rdms as vr

# ---------------------------------------------------------------------------
# Small synthetic dataset constants (used to patch module-level values)
# ---------------------------------------------------------------------------
_N = 6
_LEN = _N * (_N - 1) // 2  # 15


@pytest.fixture()
def patched_results(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR to a temp dir and shrink expected N/length so tests
    don't need a real 725-image dataset. validate_rdms.py imports RESULTS_DIR,
    _EXPECTED_LEN, and _EXPECTED_N directly from common, binding its own
    module-level names -- both modules' copies must be patched."""
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(common, "_EXPECTED_LEN", _LEN)
    monkeypatch.setattr(common, "_EXPECTED_N", _N)
    monkeypatch.setattr(vr, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)
    monkeypatch.setattr(vr, "_N", _N)
    return tmp_path


def _valid_condensed() -> np.ndarray:
    """A valid, small condensed distance vector: symmetric by construction via
    squareform, all-positive, no NaN/Inf."""
    points = np.random.rand(_N, 2)
    return pdist(points) + 0.1


# ---------------------------------------------------------------------------
# check_universal
# ---------------------------------------------------------------------------

class TestCheckUniversal:
    def test_valid_condensed_array_passes(self, monkeypatch):
        monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)
        monkeypatch.setattr(vr, "_N", _N)
        vr.check_universal("test", _valid_condensed())  # should not raise

    def test_wrong_length_raises(self, monkeypatch):
        monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)
        with pytest.raises(vr.CheckFailed, match="expected length"):
            vr.check_universal("test", np.abs(np.random.rand(_LEN - 1)))

    def test_nan_raises(self, monkeypatch):
        monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)
        monkeypatch.setattr(vr, "_N", _N)
        d = _valid_condensed()
        d[0] = np.nan
        with pytest.raises(vr.CheckFailed, match="NaN"):
            vr.check_universal("test", d)

    def test_inf_raises(self, monkeypatch):
        monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)
        monkeypatch.setattr(vr, "_N", _N)
        d = _valid_condensed()
        d[0] = np.inf
        with pytest.raises(vr.CheckFailed, match="Inf"):
            vr.check_universal("test", d)

    def test_negative_value_raises(self, monkeypatch):
        monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)
        monkeypatch.setattr(vr, "_N", _N)
        d = _valid_condensed()
        d[0] = -1.0
        with pytest.raises(vr.CheckFailed, match="negative"):
            vr.check_universal("test", d)


# ---------------------------------------------------------------------------
# check_sens / check_sens_correlation
# ---------------------------------------------------------------------------

class TestCheckSens:
    def test_positive_distances_pass(self):
        vr.check_sens(np.array([0.0, 1.0, 2.0]))  # should not raise

    def test_all_zero_raises(self):
        with pytest.raises(vr.CheckFailed, match="zero"):
            vr.check_sens(np.zeros(5))


class TestCheckSensCorrelation:
    def test_correlated_passes(self):
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        vr.check_sens_correlation(d, d)  # rho == 1.0, should not raise

    def test_anticorrelated_raises(self):
        d_pre = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        d_post = d_pre[::-1]
        with pytest.raises(vr.CheckFailed, match="Spearman rho"):
            vr.check_sens_correlation(d_pre, d_post)


# ---------------------------------------------------------------------------
# check_sem_km
# ---------------------------------------------------------------------------

class TestCheckSemKm:
    def test_valid_integer_distances_pass(self):
        vr.check_sem_km(np.array([1.0, 2.0, 3.0, 4.0]))  # should not raise

    def test_non_integer_raises(self):
        with pytest.raises(vr.CheckFailed, match="not all integers"):
            vr.check_sem_km(np.array([1.0, 2.5, 3.0]))

    def test_min_below_one_raises(self):
        with pytest.raises(vr.CheckFailed, match="minimum off-diagonal KM distance"):
            vr.check_sem_km(np.array([0.0, 2.0, 3.0]))

    def test_max_above_bound_raises(self):
        with pytest.raises(vr.CheckFailed, match="max KM distance"):
            vr.check_sem_km(np.array([1.0, 2.0, 20.0]))


# ---------------------------------------------------------------------------
# check_sem_wn
# ---------------------------------------------------------------------------

class TestCheckSemWn:
    def test_valid_range_and_low_fallback_fraction_passes(self):
        d = np.concatenate([np.full(19, 5.0), np.array([vr._WN_FALLBACK])])
        vr.check_sem_wn(d)  # should not raise

    def test_max_above_fallback_raises(self):
        with pytest.raises(vr.CheckFailed, match="max WN distance"):
            vr.check_sem_wn(np.array([1.0, vr._WN_FALLBACK + 5.0]))

    def test_high_fallback_fraction_raises(self):
        d = np.concatenate([np.full(5, vr._WN_FALLBACK), np.full(5, 1.0)])
        with pytest.raises(vr.CheckFailed, match="use fallback distance"):
            vr.check_sem_wn(d)


# ---------------------------------------------------------------------------
# check_clip / check_clip_correlation
# ---------------------------------------------------------------------------

class TestCheckClip:
    def test_valid_range_passes(self):
        vr.check_clip(np.array([0.0, 1.0, 2.0]))  # should not raise

    def test_above_bound_raises(self):
        with pytest.raises(vr.CheckFailed, match="max CLIP cosine distance"):
            vr.check_clip(np.array([1.0, 2.5]))


class TestCheckClipCorrelation:
    def test_correlated_passes(self):
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        vr.check_clip_correlation(d, d)  # rho == 1.0, should not raise

    def test_anticorrelated_raises(self):
        d_pre = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        d_post = d_pre[::-1]
        with pytest.raises(vr.CheckFailed, match="Spearman rho"):
            vr.check_clip_correlation(d_pre, d_post)


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_absent_rdm_is_skipped_not_failed(self, patched_results):
        common.save_rdm("sens_pre", _valid_condensed(), metric="euclidean", source="tests")
        # "sens_post" is never saved -> should be skipped, not counted as a failure
        assert vr.run_all_checks(names=["sens_pre", "sens_post"]) is True

    def test_load_error_surfaces_as_failure(self, patched_results):
        # Wrong-length file bypasses save_rdm(); load_rdm() raises ValueError,
        # which _try_load() converts into a load error rather than a skip.
        np.save(patched_results / "D_sens_pre.npy", np.random.rand(_LEN + 5))
        assert vr.run_all_checks(names=["sens_pre"]) is False

    def test_failing_check_returns_false(self, patched_results):
        common.save_rdm("sens_pre", np.zeros(_LEN), metric="euclidean", source="tests")
        assert vr.run_all_checks(names=["sens_pre"]) is False

    def test_all_present_and_passing_returns_true(self, patched_results):
        # sens_pre/sens_post must correlate (check_sens_correlation) as well as
        # individually pass check_sens -- reuse the same vector for both.
        d = _valid_condensed()
        common.save_rdm("sens_pre", d, metric="euclidean", source="tests")
        common.save_rdm("sens_post", d, metric="euclidean", source="tests")
        assert vr.run_all_checks(names=["sens_pre", "sens_post"]) is True
