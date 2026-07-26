"""Tests for pilot ingestion + calibration.

Session/CSV loading is delegated to ``analysis.utils.parser`` (tested there); these tests exercise
this module's own logic - reducing a parser-style tidy ``trials`` frame to a ``PilotSubject``, the
observables, and the calibration fit - on hand-built DataFrames, with no real pilot-data dependency.
"""
import json

import numpy as np
import pandas as pd
import pytest

from SpAM_Simulations import pilot
from SpAM_Simulations.experiment import _condensed_pair_indices
from SpAM_Simulations.simulation import build_ground_truth_embeddings
from SpAM_Simulations.task_v3_experiment import (
    simulate_task_v3_experiment, TaskV3ExperimentParameters,
)

MANIFEST_IMAGES = ["a.png", "b.png", "c.png", "d.png", "e.png"]  # N=5 -> 10 pairs


def _write_manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"images": MANIFEST_IMAGES, "practice_images": [], "catch_images": []}))
    return str(p)


def _pw_json(images, dists):
    """A trial's pairwise_distances JSON string (parser keeps this column as-is)."""
    return json.dumps([
        {"src1": f"./images/pre_shine/{images[i]}", "src2": f"./images/pre_shine/{images[j]}", "distance": d}
        for (i, j), d in dists.items()
    ])


def _trials_df(trials, participant="P", version=3.0):
    """Build a parser-style `trials` frame for one participant.

    `trials`: list of {images, dists, is_repeat?, repeat_of?}; `repeat_of` is the 1-based trial number
    of the original (matching `repeat_of_trial_number`).
    """
    recs = []
    for ti, t in enumerate(trials, start=1):
        recs.append({
            "participant_id": participant, "task_version": version, "trial_number": ti,
            "is_trial_repeat": bool(t.get("is_repeat", False)),
            "repeat_of_trial_number": float(t["repeat_of"]) if t.get("repeat_of") is not None else np.nan,
            "qc_flag": bool(t.get("qc", False)),
            "pairwise_distances": _pw_json(t["images"], t["dists"]),
        })
    return pd.DataFrame(recs)


def _subject(tmp_path, trials, **kw):
    _, rel2idx = pilot.load_manifest(_write_manifest(tmp_path))
    return pilot.subject_from_trials(_trials_df(trials, **kw), rel2idx)


def _cond(i, j, n=5):
    return int(_condensed_pair_indices(np.array([i]), np.array([j]), n)[0])


# --------------------------------------------------------------------- manifest / parsing
def test_load_manifest_and_src_mapping(tmp_path):
    images, rel2idx = pilot.load_manifest(_write_manifest(tmp_path))
    assert images == MANIFEST_IMAGES and rel2idx["c.png"] == 2
    assert pilot._src_to_relpath("./images/post_shine/c.png") == "c.png"


def test_subject_distances_and_counts(tmp_path):
    subj = _subject(tmp_path, [{"images": ["a.png", "b.png", "c.png"],
                                "dists": {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}}])
    assert subj.distances[_cond(0, 1)] == pytest.approx(0.1)
    assert subj.distances[_cond(1, 2)] == pytest.approx(0.7)
    assert np.isnan(subj.distances[_cond(3, 4)])  # unobserved pair
    assert subj.num_observed_pairs() == 3
    assert subj.task_version == 3.0 and subj.participant_id == "P"


def test_repeat_trial_averaged_and_aligned(tmp_path):
    base = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 0.9}
    rep = {(0, 1): 0.4, (0, 2): 0.5, (1, 2): 0.7}
    subj = _subject(tmp_path, [
        {"images": ["a.png", "b.png", "c.png"], "dists": base},
        {"images": ["a.png", "b.png", "c.png"], "dists": rep, "is_repeat": True, "repeat_of": 1},
    ])
    assert subj.distances[_cond(0, 1)] == pytest.approx(0.3)  # (a,b) seen twice -> mean(0.2, 0.4)
    assert len(subj.retest_pairs) == 1
    orig, rp = subj.retest_pairs[0]
    assert len(orig) == 3 and len(rp) == 3


# --------------------------------------------------------------------- observables
def test_within_subject_test_retest_value(tmp_path):
    same = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 0.9}
    subj = _subject(tmp_path, [
        {"images": ["a.png", "b.png", "c.png"], "dists": same},
        {"images": ["a.png", "b.png", "c.png"], "dists": same, "is_repeat": True, "repeat_of": 1},
    ])
    assert pilot.within_subject_test_retest(subj) == pytest.approx(1.0)  # identical -> Spearman 1


def test_between_subject_agreement(tmp_path):
    d1 = {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}
    d2 = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 0.9}  # same rank order -> agreement 1.0
    s1 = _subject(tmp_path, [{"images": ["a.png", "b.png", "c.png"], "dists": d1}], participant="A")
    s2 = _subject(tmp_path, [{"images": ["a.png", "b.png", "c.png"], "dists": d2}], participant="B")
    out = pilot.between_subject_agreement(pilot.stack_distances([s1, s2]), min_overlap=2)
    assert out["mean_agreement"] == pytest.approx(1.0)
    assert out["n_dyads"] == 1 and out["median_overlap"] == 3


# --------------------------------------------------------------------- load_pilot_subjects (parser-backed)
def test_load_subjects_filters_version_and_qc(tmp_path, monkeypatch):
    man = _write_manifest(tmp_path)
    good = {"images": ["a.png", "b.png", "c.png"], "dists": {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}}
    flagged = {**good, "qc": True}
    trials = pd.concat([
        _trials_df([good], participant="A", version=3.0),
        _trials_df([good], participant="B", version=2.0),
        _trials_df([flagged], participant="C", version=3.0),  # 100% flagged
    ], ignore_index=True)
    # stub the parser loader: this module must NOT re-read CSVs, only reduce the tidy trials frame
    monkeypatch.setattr(pilot, "load_pilot_data", lambda d: {"trials": trials})
    assert len(pilot.load_pilot_subjects("ignored", man)) == 3
    assert len(pilot.load_pilot_subjects("ignored", man, versions=[3.0])) == 2
    assert len(pilot.load_pilot_subjects("ignored", man, apply_qc=True)) == 2  # drops the flagged one


def test_load_subjects_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "load_pilot_data", lambda d: {"trials": pd.DataFrame()})
    assert pilot.load_pilot_subjects("ignored", _write_manifest(tmp_path)) == []


def test_aggregate_raises_on_disconnected_graph(tmp_path):
    # subject only ever touches images {a,b,c}; d,e never observed -> graph disconnected
    s = _subject(tmp_path, [{"images": ["a.png", "b.png", "c.png"],
                             "dists": {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}}])
    with pytest.raises(RuntimeError, match="connected components"):
        pilot.pilot_aggregate([s])


# --------------------------------------------------------------------- simulation hook + calibration
def test_simulate_returns_per_subject_matching_nan_pattern():
    emb = build_ground_truth_embeddings(80, 4, seed=1)
    params = TaskV3ExperimentParameters(6, 8, 8, 0.4, 1, 0.25, 0.2)
    _, res, per_subject = simulate_task_v3_experiment(params, emb, np.random.default_rng(0),
                                                      verbose=False, return_per_subject=True)
    assert per_subject.shape == (6, res.distances.shape[0])
    out = pilot.between_subject_agreement(per_subject, min_overlap=5)
    assert -1.0 <= out["mean_agreement"] <= 1.0 and out["n_dyads"] > 0


def test_simulated_targets_are_monotone():
    """Validity of the fit direction: more noise -> lower test-retest; more dispersion -> lower agreement."""
    emb = build_ground_truth_embeddings(120, 5, seed=2)
    kw = dict(num_subjects=10, trials_per_subject=8, images_per_trial=10,
              frac_trials_repeated=0.25, reps=3, seed=0, min_overlap=5)
    tr_lo, _ = pilot._simulated_targets(emb, noise_scale=0.2, dispersion=0.0, **kw)
    tr_hi, _ = pilot._simulated_targets(emb, noise_scale=1.5, dispersion=0.0, **kw)
    assert tr_lo > tr_hi  # noise lowers test-retest
    _, agr_lo = pilot._simulated_targets(emb, noise_scale=0.5, dispersion=0.0, **kw)
    _, agr_hi = pilot._simulated_targets(emb, noise_scale=0.5, dispersion=0.8, **kw)
    assert agr_lo > agr_hi  # dispersion lowers between-subject agreement


def _fake_subjects(tmp_path):
    """2 v3 subjects (with repeats) + 1 v2 subject, all observing images a,b,c."""
    same = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 0.9}
    v3 = [_subject(tmp_path, [{"images": ["a.png", "b.png", "c.png"], "dists": same},
                              {"images": ["a.png", "b.png", "c.png"], "dists": same,
                               "is_repeat": True, "repeat_of": 1}], participant=p, version=3.0)
          for p in ("A", "B")]
    other = [_subject(tmp_path, [{"images": ["a.png", "b.png", "c.png"], "dists": same}],
                      participant="C", version=2.0)]
    return v3, other


def _stub_gt_and_fit(monkeypatch, captured):
    """Patch build_gt_from_pilot + _calibrate (no R / no heavy sim); capture the fit inputs."""
    monkeypatch.setattr(pilot, "build_gt_from_pilot",
                        lambda subs, n_dims=None, method="smacof":
                        (np.zeros((5, 3), np.float32), {"n_dims": 3, "method": method, "observed_frac": 1.0}))

    def _fake_calibrate(coords, num_subjects, target_tr, target_agr, *, reps=5, min_overlap=30, **kw):
        captured.update(num_subjects=num_subjects, target_tr=target_tr, target_agr=target_agr, reps=reps)
        return {"subjects_noise_scale": 1.0, "perspective_dispersion": 0.2, "subjects_noise_df": 1,
                "pilot_test_retest": target_tr, "pilot_between_agreement": target_agr,
                "simulated_test_retest": 0.3, "simulated_between_agreement": 0.2, "num_subjects": num_subjects}
    monkeypatch.setattr(pilot, "_calibrate", _fake_calibrate)


def test_calibrate_params_from_pilot_uses_v3_for_retest_and_all_for_agreement(tmp_path, monkeypatch):
    """test-retest target from v3 only; between-subject agreement pooled over ALL subjects."""
    v3, other = _fake_subjects(tmp_path)
    captured = {}
    monkeypatch.setattr(pilot, "load_pilot_subjects", lambda d, m: v3 + other)
    _stub_gt_and_fit(monkeypatch, captured)
    # spy on the agreement call to confirm it receives all 3 subjects, not just the 2 v3 ones
    seen = {}

    def _spy_agr(dists, min_overlap=30):
        seen["n_rows"] = dists.shape[0]
        return {"mean_agreement": 0.2, "sem_agreement": 0.0, "n_dyads": 3, "median_overlap": 3}
    monkeypatch.setattr(pilot, "between_subject_agreement", _spy_agr)

    coords, fit, info = pilot.calibrate_params_from_pilot("d", "m", gt_method="classical", reps=3, verbose=False)
    assert coords.shape == (5, 3) and info["n_dims"] == 3
    assert seen["n_rows"] == 3            # agreement over all subjects (2 v3 + 1 v2)
    assert captured["num_subjects"] == 2  # the fit's matched sim uses the 2 v3 subjects
    assert captured["reps"] == 3 and fit["perspective_dispersion"] == 0.2


def test_calibrate_params_from_pilot_saves_artifacts(tmp_path, monkeypatch):
    import json
    v3, other = _fake_subjects(tmp_path)
    monkeypatch.setattr(pilot, "load_pilot_subjects", lambda d, m: v3 + other)
    _stub_gt_and_fit(monkeypatch, {})
    gt_path = tmp_path / "gt.npy"; params_path = tmp_path / "params.json"
    pilot.calibrate_params_from_pilot("d", "m", gt_method="classical",
                                      save_gt=str(gt_path), save_params=str(params_path), verbose=False)
    assert np.load(gt_path).shape == (5, 3)
    saved = json.loads(params_path.read_text())
    assert saved["subjects_noise_scale"] == 1.0 and saved["n_dims"] == 3 and saved["gt_method"] == "classical"


def test_calibrate_params_from_pilot_raises_without_v3(tmp_path, monkeypatch):
    only_v2 = [_subject(tmp_path, [{"images": ["a.png", "b.png", "c.png"],
                                    "dists": {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}}], version=2.0)]
    monkeypatch.setattr(pilot, "load_pilot_subjects", lambda d, m: only_v2)
    with pytest.raises(SystemExit, match="no v3.0 subjects"):
        pilot.calibrate_params_from_pilot("d", "m", verbose=False)


def test_calibrate_recovers_known_parameters():
    """Self-consistency: simulate a 'pilot' cohort at known (noise, dispersion), then recover them."""
    emb = build_ground_truth_embeddings(400, 5, seed=3)  # >= t_distinct*k = 17*20 = 340
    true_noise, true_disp = 0.6, 0.4
    kw = dict(num_subjects=11, trials_per_subject=20, images_per_trial=20,
              frac_trials_repeated=0.15, reps=6, seed=11, min_overlap=10)
    target_tr, target_agr = pilot._simulated_targets(emb, true_noise, true_disp, **kw)

    fit_noise = pilot._fit_1d(target_tr,
                              lambda x: pilot._simulated_targets(emb, x, 0.0, **kw)[0],
                              np.round(np.arange(0.2, 1.21, 0.2), 2))
    fit_disp = pilot._fit_1d(target_agr,
                             lambda x: pilot._simulated_targets(emb, fit_noise, x, **kw)[1],
                             np.round(np.arange(0.0, 0.81, 0.2), 2))
    assert abs(fit_noise - true_noise) <= 0.25  # within ~1 grid step
    assert abs(fit_disp - true_disp) <= 0.25


class TestDispersionRefit:
    """Dispersion must be re-fitted whenever the noise POPULATION changes, not just its mean."""

    GT = None

    def _gt(self):
        from SpAM_Simulations.simulation import build_ground_truth_embeddings
        if TestDispersionRefit.GT is None:
            TestDispersionRefit.GT = build_ground_truth_embeddings(90, 4, seed=2)
        return TestDispersionRefit.GT

    def test_agreement_depends_on_the_noise_shape_not_only_its_mean(self):
        """The reason a shape refit invalidates the old dispersion calibration.

        Same mean noise, same dispersion, different noise SHAPE -> different between-subject
        agreement. If this were not so, dispersion could be calibrated once and reused.
        """
        from SpAM_Simulations.pilot import _simulated_targets
        common = dict(gt_embeddings=self._gt(), noise_scale=0.8, dispersion=0.3, num_subjects=25,
                      trials_per_subject=8, images_per_trial=8, frac_trials_repeated=0.25,
                      reps=2, seed=0, min_overlap=3)
        concentrated = _simulated_targets(**common, noise_df=5, lognormal_sigma=0.15)[1]
        dispersed = _simulated_targets(**common, noise_df=5, lognormal_sigma=1.0)[1]
        assert abs(concentrated - dispersed) > 0.01

    def test_sigma_zero_keeps_the_historical_t_family_path(self):
        """Back-compat: the default routes through |t(df)|, so old calibrations still reproduce."""
        from SpAM_Simulations.pilot import _simulated_targets
        common = dict(gt_embeddings=self._gt(), noise_scale=0.8, dispersion=0.3, num_subjects=20,
                      trials_per_subject=8, images_per_trial=8, frac_trials_repeated=0.25,
                      reps=1, seed=0, min_overlap=3, noise_df=5)
        a = _simulated_targets(**common)
        b = _simulated_targets(**common, lognormal_sigma=0.0)
        assert a == b

    def test_fit_returns_a_grid_value_and_its_achieved_agreement(self):
        from SpAM_Simulations.pilot import fit_dispersion_for_agreement
        grid = (0.0, 0.2, 0.4)
        disp, ach = fit_dispersion_for_agreement(
            self._gt(), 0.5, noise_scale=0.8, noise_df=5, dispersion_grid=grid,
            num_subjects=20, trials_per_subject=8, images_per_trial=8,
            frac_trials_repeated=0.25, reps=1, min_overlap=3)
        assert disp in grid and np.isfinite(ach)
