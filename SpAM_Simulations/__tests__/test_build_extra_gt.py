"""Tests for the extra-GT builder.

It exists because the stage-1 scan selects the dimensionality at which two 20-subject halves still
agree - a floor set by sample size, not the intrinsic dimensionality - and a planning simulation
built on that floor understates required-N.
"""
import json

import numpy as np
import pytest

from SpAM_Simulations import build_extra_gt as beg


class _Subject:
    def __init__(self, distances, n_obs):
        self.distances = np.asarray(distances, dtype=np.float32)
        self.n_obs = np.asarray(n_obs, dtype=np.int32)
        self.task_version = 3.0
        self.shine_variant = "pre"
        self.participant_id = "P"


def _cohort(n_subjects=6, n_images=10, seed=0):
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(seed)
    truth = pdist(rng.normal(size=(n_images, 3)))
    return [_Subject(truth + rng.normal(0, 0.05, truth.size),
                     np.ones(truth.size, dtype=np.int32)) for _ in range(n_subjects)]


def _patch_loader(monkeypatch, subjects):
    from SpAM_Simulations import pilot
    monkeypatch.setattr(pilot, "load_pilot_subjects",
                        lambda *a, **k: subjects, raising=True)


def test_builds_and_records_an_extra_gt(monkeypatch, tmp_path):
    subs = _cohort()
    _patch_loader(monkeypatch, subs)
    out = beg.build(4, gt_dir=tmp_path, method="classical", expect_n_subjects=len(subs))
    assert out.name == "gt_pre_shine_d4.npy"
    assert np.load(out).shape == (10, 4)

    notes = json.loads((tmp_path / "extra_gts.json").read_text())
    entry = notes["built"][0]
    assert entry["n_dims"] == 4 and entry["gt_file"] == "gt_pre_shine_d4.npy"
    assert "reason" in entry and "built_utc" in entry


def test_selection_json_is_never_touched(monkeypatch, tmp_path):
    """It records what the EVIDENCE chose; the override is documented at the point of use."""
    _patch_loader(monkeypatch, _cohort())
    original = {"n_dims": 3, "gt_file": "gt_pre_shine_d3.npy"}
    (tmp_path / "selection.json").write_text(json.dumps(original))
    beg.build(5, gt_dir=tmp_path, method="classical", expect_n_subjects=6)
    assert json.loads((tmp_path / "selection.json").read_text()) == original


def test_repeated_builds_replace_rather_than_duplicate(monkeypatch, tmp_path):
    _patch_loader(monkeypatch, _cohort())
    for _ in range(2):
        beg.build(6, gt_dir=tmp_path, method="classical", expect_n_subjects=6)
    beg.build(8, gt_dir=tmp_path, method="classical", expect_n_subjects=6)
    notes = json.loads((tmp_path / "extra_gts.json").read_text())
    assert [b["n_dims"] for b in notes["built"]] == [6, 8]      # deduped and sorted


def test_a_changed_subject_set_is_refused(monkeypatch, tmp_path):
    """The extra GT must sit on the same subjects as the scan or it cannot be compared with it."""
    _patch_loader(monkeypatch, _cohort(n_subjects=5))
    with pytest.raises(SystemExit, match="expected 41"):
        beg.build(4, gt_dir=tmp_path, method="classical")


def test_cli_rejects_a_nonpositive_ndim():
    with pytest.raises(SystemExit):
        beg.main(["--ndim", "0"])
