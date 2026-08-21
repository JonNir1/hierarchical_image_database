"""Tests for the v6 ground-truth rebuild.

The rebuild adds the production subjects the analysis discards to the pilot's 41. Two things must
hold and neither is obvious from reading the code: the subject-count assertions have to fire (a GT
built on a silently different subject set is not comparable to the one the decision run was
calibrated against), and the accept/reject decision has to be ONE-SIDED, because the augmented set
has larger split-halves and is favoured for reasons unrelated to whether the added data is any good.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from SpAM_Simulations.cli import build_gt_v6 as bg

N_IMAGES = 30
N_PAIRS = N_IMAGES * (N_IMAGES - 1) // 2


def _subject(seed):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(participant_id=f"p{seed}", distances=rng.random(N_PAIRS),
                           n_obs=np.ones(N_PAIRS, dtype=np.int64))


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Stub every expensive dependency: the point is the control flow, not SMACOF."""
    calls = {"diagnostics": [], "build_gt": []}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": [f"a/b/img{i}.png" for i in range(N_IMAGES)]}))

    monkeypatch.setattr(bg, "excluded_prod_subjects",
                        lambda *a, **k: [_subject(100 + i) for i in range(8)])

    import SpAM_Simulations.empirical.gt_construction as gtc
    import SpAM_Simulations.empirical.subjects as subj
    monkeypatch.setattr(subj, "load_pilot_subjects",
                        lambda *a, **k: [_subject(i) for i in range(41)])
    monkeypatch.setattr(gtc, "is_connected", lambda s: True)
    monkeypatch.setattr(gtc, "coverage_of", lambda s: 0.3 + 0.001 * len(s))

    def fake_build_gt(subjects, ndim, method="smacof"):
        calls["build_gt"].append(len(subjects))
        return np.zeros((N_IMAGES, ndim)), {"method": method, "n_dims": ndim}

    monkeypatch.setattr(gtc, "build_gt", fake_build_gt)
    return calls, manifest, tmp_path


def _diag(spearman, sem=0.01):
    return {"n_subjects": 0, "coverage": 0.3, "split_half_spearman": spearman,
            "split_half_spearman_sem": sem, "split_half_procrustes_m2": 0.5,
            "split_half_topk_jaccard": 0.2, "max_iters_rate": 0.0,
            "mean_noise_ceiling_full": 0.4, "half_size": 20, "discard_rate": 0.1,
            "coverage_gap_frac": 0.001}


def _run(patched, monkeypatch, before, after, extra_args=()):
    calls, manifest, tmp_path = patched
    seq = iter([_diag(before), _diag(after)])
    monkeypatch.setattr(bg, "_diagnostics", lambda *a, **k: next(seq))
    rc = bg.main(["--gt-dir", str(tmp_path / "gt"), "--manifest", str(manifest),
                  "--data-dir", "unused", "--ndim", "4", *extra_args])
    decision = json.loads((tmp_path / "gt" / "gt_v6_decision.json").read_text())
    return rc, decision, calls


class TestDecision:
    def test_an_improvement_is_accepted(self, patched, monkeypatch):
        _, decision, calls = _run(patched, monkeypatch, before=0.30, after=0.34)
        assert decision["accepted"] is True
        assert decision["n_subjects_used"] == 49
        assert calls["build_gt"] == [49]
        assert decision["gt_file"] == "gt_pre_shine_v6_d4.npy"

    def test_a_small_loss_inside_the_margin_is_still_accepted(self, patched, monkeypatch):
        """One-sided by design: a drop smaller than the draw-to-draw noise is not evidence."""
        _, decision, _ = _run(patched, monkeypatch, before=0.30, after=0.29)
        assert decision["accepted"] is True

    def test_a_real_loss_is_rejected_and_falls_back_to_the_pilot(self, patched, monkeypatch):
        _, decision, calls = _run(patched, monkeypatch, before=0.30, after=0.20)
        assert decision["accepted"] is False
        assert decision["n_subjects_used"] == 41
        assert calls["build_gt"] == [41]
        assert decision["gt_file"] == "gt_pre_shine_pilot_only_d4.npy"

    def test_rejection_still_exits_zero(self, patched, monkeypatch):
        """A rejected rebuild is an informative outcome, not a failure: the pilot-only GT is
        written and `accepted` carries the verdict."""
        rc, _, _ = _run(patched, monkeypatch, before=0.30, after=0.20)
        assert rc == 0

    def test_the_margin_scales_with_the_measured_noise(self, patched, monkeypatch):
        _, decision, _ = _run(patched, monkeypatch, before=0.30, after=0.29)
        assert decision["margin"] == pytest.approx(2.0 * 0.01)

    def test_skip_comparison_builds_without_scoring(self, patched, monkeypatch):
        calls, manifest, tmp_path = patched
        monkeypatch.setattr(bg, "_diagnostics",
                            lambda *a, **k: pytest.fail("should not have scored"))
        bg.main(["--gt-dir", str(tmp_path / "gt"), "--manifest", str(manifest),
                 "--data-dir", "unused", "--ndim", "4", "--skip-comparison"])
        decision = json.loads((tmp_path / "gt" / "gt_v6_decision.json").read_text())
        assert decision["comparison"] == "skipped"
        assert decision["accepted"] is True


class TestSubjectCountAssertions:
    def test_an_unexpected_pilot_count_aborts(self, patched, monkeypatch):
        with pytest.raises(SystemExit, match="pre-SHINE pilot subjects"):
            _run(patched, monkeypatch, 0.3, 0.3, extra_args=("--expect-pilot", "40"))

    def test_an_unexpected_excluded_count_aborts(self, patched, monkeypatch):
        """This one GROWS as collection continues, so it must be raised deliberately: a GT rebuilt
        on a different subject set is not comparable to the one the run was calibrated against."""
        with pytest.raises(SystemExit, match="discarded pre-SHINE production subjects"):
            _run(patched, monkeypatch, 0.3, 0.3, extra_args=("--expect-excluded", "12"))
