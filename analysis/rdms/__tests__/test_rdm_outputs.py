"""
Output-validation tests for the built RDMs in analysis/results/rdms/.

These tests operate on the *real generated outputs*, not synthetic data.
Each test is skipped automatically when the corresponding .npy file is absent
(i.e. the RDM has not yet been built), so this suite is safe to run in CI
even if only a subset of RDMs has been generated.

Run locally after building all (or some) RDMs:
    pytest analysis/rdms/__tests__/test_rdm_outputs.py -v

For CI with shared artifacts: download the .npy files and metadata.json into
analysis/results/rdms/ before running pytest, then include this file in the
test run.
"""
from __future__ import annotations

import pytest

from analysis.rdms.validate_rdms import (
    _try_load,
    check_clip,
    check_clip_correlation,
    check_sem_km,
    check_sem_wn,
    check_sens,
    check_sens_correlation,
    check_universal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(name: str):
    """Load an RDM or skip the test if it is not present."""
    d, err = _try_load(name)
    if err is not None:
        pytest.fail(f"D_{name}.npy exists but failed to load: {err}")
    if d is None:
        pytest.skip(f"D_{name}.npy not found — run build_all first")
    return d


# ---------------------------------------------------------------------------
# Universal checks (parametrised over every RDM)
# ---------------------------------------------------------------------------

_ALL_NAMES = ["sens_pre", "sens_post", "sem_km", "sem_wn", "clip_pre", "clip_post"]


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_universal(name):
    """Shape, finiteness, non-negativity, symmetry, zero diagonal."""
    d = _require(name)
    check_universal(name, d)


# ---------------------------------------------------------------------------
# Sensory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["sens_pre", "sens_post"])
def test_sensory_distances_positive(name):
    """Pixel Euclidean distances should be > 0 (non-identical images)."""
    d = _require(name)
    check_sens(d)


def test_sensory_pre_post_correlation():
    """Pre- and post-SHINE sensory RDMs: Spearman rho > 0.9."""
    d_pre = _require("sens_pre")
    d_post = _require("sens_post")
    check_sens_correlation(d_pre, d_post)


# ---------------------------------------------------------------------------
# Semantic KM
# ---------------------------------------------------------------------------

def test_sem_km_integer_distances():
    """KM distances are positive integers bounded by 2 * hierarchy depth."""
    d = _require("sem_km")
    check_sem_km(d)


# ---------------------------------------------------------------------------
# Semantic WordNet
# ---------------------------------------------------------------------------

def test_sem_wn_range_and_fallback():
    """WN distances in [0, 30]; fewer than 10% of pairs hit the fallback."""
    d = _require("sem_wn")
    check_sem_wn(d)


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["clip_pre", "clip_post"])
def test_clip_cosine_range(name):
    """CLIP cosine distances lie in [0, 2]."""
    d = _require(name)
    check_clip(d)


def test_clip_pre_post_correlation():
    """Pre- and post-SHINE CLIP RDMs: Spearman rho > 0.9."""
    d_pre = _require("clip_pre")
    d_post = _require("clip_post")
    check_clip_correlation(d_pre, d_post)
