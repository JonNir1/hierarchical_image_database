"""Tests for analysis.build_rdms.validate_rdms: the pure numpy/scipy invariant checks
and the run_all_checks orchestration (load/skip/failure aggregation)."""
from __future__ import annotations

import numpy as np
import pytest
from _rdm_helpers import patch_common_results
from scipy.spatial.distance import pdist

import analysis.build_rdms.common as common
import analysis.build_rdms.validate_rdms as vr

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
    patch_common_results(monkeypatch, tmp_path, _N)
    monkeypatch.setattr(vr, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)
    monkeypatch.setattr(vr, "_N", _N)
    # run_all_checks opens with the manifest phase, which reads the real
    # manifest and the real image directories. Stub it out so these tests stay
    # unit-level; it has dedicated coverage in TestCheckManifestMatchesDisk.
    monkeypatch.setattr(vr, "check_manifest_matches_disk", lambda: None)
    return tmp_path


def _valid_condensed() -> np.ndarray:
    """A valid, small condensed distance vector: symmetric by construction via
    squareform, all-positive, no NaN/Inf."""
    points = np.random.rand(_N, 2)
    return pdist(points) + 0.1


# ---------------------------------------------------------------------------
# check_universal
# ---------------------------------------------------------------------------

def _with_first_value(value: float) -> np.ndarray:
    d = _valid_condensed()
    d[0] = value
    return d


@pytest.fixture()
def small_n(monkeypatch):
    monkeypatch.setattr(vr, "_EXPECTED_LEN", _LEN)
    monkeypatch.setattr(vr, "_N", _N)


class TestCheckUniversal:
    def test_valid_condensed_array_passes(self, small_n):
        vr.check_universal("test", _valid_condensed())  # should not raise

    @pytest.mark.parametrize("make_array, match", [
        pytest.param(lambda: np.abs(np.random.rand(_LEN - 1)), "expected length", id="wrong_length"),
        pytest.param(lambda: _with_first_value(np.nan), "NaN", id="nan"),
        pytest.param(lambda: _with_first_value(np.inf), "Inf", id="inf"),
        pytest.param(lambda: _with_first_value(-1.0), "negative", id="negative"),
    ])
    def test_invalid_array_raises(self, small_n, make_array, match):
        with pytest.raises(vr.CheckFailed, match=match):
            vr.check_universal("test", make_array())


# ---------------------------------------------------------------------------
# Single-array checks: check_sens, check_sem_km, check_sem_wn, check_clip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("check, d", [
    pytest.param(vr.check_sens, np.array([0.0, 1.0, 2.0]), id="sens"),
    pytest.param(vr.check_sem_km, np.array([1.0, 2.0, 3.0, 4.0]), id="sem_km_integers"),
    pytest.param(vr.check_sem_wn, np.concatenate([np.full(19, 5.0), np.array([vr._WN_FALLBACK])]),
                 id="sem_wn_low_fallback_fraction"),
    pytest.param(vr.check_clip, np.array([0.0, 1.0, 2.0]), id="clip"),
])
def test_single_array_check_passes(check, d):
    check(d)  # should not raise


@pytest.mark.parametrize("check, d, match", [
    pytest.param(vr.check_sens, np.zeros(5), "zero", id="sens_all_zero"),
    pytest.param(vr.check_sem_km, np.array([1.0, 2.5, 3.0]), "not all integers", id="sem_km_non_integer"),
    pytest.param(vr.check_sem_km, np.array([0.0, 2.0, 3.0]), "minimum off-diagonal KM distance",
                 id="sem_km_min_below_one"),
    pytest.param(vr.check_sem_km, np.array([1.0, 2.0, 20.0]), "max KM distance", id="sem_km_max_above_bound"),
    pytest.param(vr.check_sem_wn, np.array([1.0, vr._WN_FALLBACK + 5.0]), "max WN distance",
                 id="sem_wn_max_above_fallback"),
    pytest.param(vr.check_sem_wn, np.concatenate([np.full(5, vr._WN_FALLBACK), np.full(5, 1.0)]),
                 "use fallback distance", id="sem_wn_high_fallback_fraction"),
    pytest.param(vr.check_clip, np.array([1.0, 2.5]), "max CLIP cosine distance", id="clip_above_bound"),
])
def test_single_array_check_raises(check, d, match):
    with pytest.raises(vr.CheckFailed, match=match):
        check(d)


# ---------------------------------------------------------------------------
# check_sens_correlation / check_clip_correlation (identical contract)
# ---------------------------------------------------------------------------

_CORRELATION_CHECKS = [vr.check_sens_correlation, vr.check_clip_correlation]
_RAMP = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


@pytest.mark.parametrize("check", _CORRELATION_CHECKS)
def test_correlated_pair_passes(check):
    check(_RAMP, _RAMP)  # rho == 1.0, should not raise


@pytest.mark.parametrize("check", _CORRELATION_CHECKS)
def test_anticorrelated_pair_raises(check):
    with pytest.raises(vr.CheckFailed, match="Spearman rho"):
        check(_RAMP, _RAMP[::-1])

# ---------------------------------------------------------------------------
# check_manifest_matches_disk
# ---------------------------------------------------------------------------

_MANIFEST_HEADER = "curated_path,curated_filename,category,wn_synset_name\n"
_TEST_PATHS = [
    r"animate\human\face\caucasian\male1.png",
    r"animate\human\face\hispanic\male1.png",
    r"inanimate\tool\hammer1.png",
]


def _write_manifest(path, curated_paths: list[str]) -> None:
    rows = "".join(
        f"{p},{p.rsplit(chr(92), 1)[-1]},HF,man.n.01\n" for p in curated_paths
    )
    path.write_text(_MANIFEST_HEADER + rows, encoding="utf-8")


@pytest.fixture()
def manifest_env(tmp_path, monkeypatch):
    """Build a 3-image dataset on disk (both SHINE variants) plus a matching
    manifest, and point common/validate_rdms at them. Returns the manifest path
    so individual tests can rewrite it into a broken state."""
    images_root = tmp_path / "images"
    for variant in ("pre_shine", "post_shine"):
        for rel in _TEST_PATHS:
            f = images_root / variant / rel.replace("\\", "/")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"")  # content is irrelevant; only the path set is checked

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, _TEST_PATHS)

    monkeypatch.setattr(common, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(vr, "IMAGES_ROOT", images_root)
    monkeypatch.setattr(vr, "_EXPECTED_N", len(_TEST_PATHS))
    common.order_hash.cache_clear()
    return manifest_path


class TestCheckManifestMatchesDisk:
    def test_matching_manifest_passes(self, manifest_env):
        vr.check_manifest_matches_disk()  # should not raise

    def test_duplicate_path_raises(self, manifest_env):
        # The dbd2b3e curation error: a row keeps a stale directory component
        # and collides with an existing row. Row count is unchanged and every
        # path still resolves to a real file, so only uniqueness catches it.
        _write_manifest(manifest_env, [
            r"animate\human\face\caucasian\male1.png",
            r"animate\human\face\caucasian\male1.png",
            r"inanimate\tool\hammer1.png",
        ])
        with pytest.raises(vr.CheckFailed, match="duplicated curated_path"):
            vr.check_manifest_matches_disk()

    def test_wrong_row_count_raises(self, manifest_env):
        _write_manifest(manifest_env, _TEST_PATHS[:-1])
        with pytest.raises(vr.CheckFailed, match="rows, expected"):
            vr.check_manifest_matches_disk()

    def test_path_with_no_image_on_disk_raises(self, manifest_env):
        _write_manifest(manifest_env, [
            r"animate\human\face\caucasian\male1.png",
            r"animate\human\face\hispanic\male99.png",  # not on disk
            r"inanimate\tool\hammer1.png",
        ])
        with pytest.raises(vr.CheckFailed, match="no image on disk"):
            vr.check_manifest_matches_disk()

    def test_post_shine_divergence_raises(self, manifest_env):
        # pre_shine matches but post_shine does not: pre/post correspondence is
        # a filename match, so a variant-only gap must still fail.
        (vr.IMAGES_ROOT / "post_shine" / "inanimate" / "tool" / "hammer1.png").unlink()
        with pytest.raises(vr.CheckFailed, match="post_shine"):
            vr.check_manifest_matches_disk()


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
