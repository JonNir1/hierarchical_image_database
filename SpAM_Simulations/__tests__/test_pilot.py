"""Tests for pilot ingestion + calibration (synthetic CSVs only - no real pilot data dependency)."""
import json

import numpy as np
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


def _pairwise(images, dists):
    """JSON for one trial: `images` relpaths, `dists` a {(i,j): distance} dict over their indices."""
    rows = []
    for (i, j), d in dists.items():
        rows.append({"src1": f"./images/pre_shine/{images[i]}",
                     "src2": f"./images/pre_shine/{images[j]}", "distance": d})
    return json.dumps(rows)


def _write_session(tmp_path, name, participant, trials, version="3.0"):
    """`trials`: list of dicts {images, dists, is_repeat?, repeat_of?}. Writes a minimal session CSV."""
    import csv
    cols = ["trial_type", "trial_index", "participant_id", "task_version",
            "is_trial_repeat", "repeat_of_trial_index", "qc_flag", "pairwise_distances"]
    p = tmp_path / name
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for ti, t in enumerate(trials):
            w.writerow({
                "trial_type": f"trial_{ti + 1}", "trial_index": ti,
                "participant_id": participant, "task_version": version,
                "is_trial_repeat": str(t.get("is_repeat", False)).lower(),
                "repeat_of_trial_index": t.get("repeat_of", "null"),
                "qc_flag": str(t.get("qc", False)).lower(),
                "pairwise_distances": _pairwise(t["images"], t["dists"]),
            })
    return str(p)


def _cond(i, j, n=5):
    return int(_condensed_pair_indices(np.array([i]), np.array([j]), n)[0])


# --------------------------------------------------------------------- manifest / parsing
def test_load_manifest_and_src_mapping(tmp_path):
    images, rel2idx = pilot.load_manifest(_write_manifest(tmp_path))
    assert images == MANIFEST_IMAGES and rel2idx["c.png"] == 2
    assert pilot._src_to_relpath("./images/post_shine/c.png") == "c.png"


def test_load_subject_distances_and_counts(tmp_path):
    man = _write_manifest(tmp_path)
    _, rel2idx = pilot.load_manifest(man)
    # one trial over images a,b,c with known distances
    trials = [{"images": ["a.png", "b.png", "c.png"], "dists": {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}}]
    path = _write_session(tmp_path, "s1.csv", "P1", trials)
    subj = pilot.load_pilot_subject(path, rel2idx)
    assert subj.distances[_cond(0, 1)] == pytest.approx(0.1)
    assert subj.distances[_cond(1, 2)] == pytest.approx(0.7)
    assert np.isnan(subj.distances[_cond(3, 4)])  # unobserved pair
    assert subj.num_observed_pairs() == 3


def test_repeat_trial_averaged_and_aligned(tmp_path):
    man = _write_manifest(tmp_path)
    _, rel2idx = pilot.load_manifest(man)
    base = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 0.9}
    rep = {(0, 1): 0.4, (0, 2): 0.5, (1, 2): 0.7}
    trials = [
        {"images": ["a.png", "b.png", "c.png"], "dists": base},
        {"images": ["a.png", "b.png", "c.png"], "dists": rep, "is_repeat": True, "repeat_of": 0},
    ]
    subj = pilot.load_pilot_subject(_write_session(tmp_path, "s.csv", "P", trials), rel2idx)
    # pair (a,b) observed twice -> mean of 0.2 and 0.4
    assert subj.distances[_cond(0, 1)] == pytest.approx(0.3)
    assert len(subj.retest_pairs) == 1
    orig, rp = subj.retest_pairs[0]
    assert len(orig) == 3 and len(rp) == 3


# --------------------------------------------------------------------- observables
def test_within_subject_test_retest_value(tmp_path):
    _, rel2idx = pilot.load_manifest(_write_manifest(tmp_path))
    # identical original/repeat -> Spearman 1.0
    same = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 0.9}
    trials = [{"images": ["a.png", "b.png", "c.png"], "dists": same},
              {"images": ["a.png", "b.png", "c.png"], "dists": same, "is_repeat": True, "repeat_of": 0}]
    subj = pilot.load_pilot_subject(_write_session(tmp_path, "s.csv", "P", trials), rel2idx)
    assert pilot.within_subject_test_retest(subj) == pytest.approx(1.0)


def test_between_subject_agreement(tmp_path):
    _, rel2idx = pilot.load_manifest(_write_manifest(tmp_path))
    d1 = {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}
    d2 = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 0.9}  # same rank order -> agreement 1.0
    s1 = pilot.load_pilot_subject(
        _write_session(tmp_path, "s1.csv", "A", [{"images": ["a.png", "b.png", "c.png"], "dists": d1}]), rel2idx)
    s2 = pilot.load_pilot_subject(
        _write_session(tmp_path, "s2.csv", "B", [{"images": ["a.png", "b.png", "c.png"], "dists": d2}]), rel2idx)
    out = pilot.between_subject_agreement(pilot.stack_distances([s1, s2]), min_overlap=2)
    assert out["mean_agreement"] == pytest.approx(1.0)
    assert out["n_dyads"] == 1 and out["median_overlap"] == 3


# --------------------------------------------------------------------- loading dir / QC / aggregate
def test_load_dir_filters_version_and_qc(tmp_path):
    man = _write_manifest(tmp_path)
    good = [{"images": ["a.png", "b.png", "c.png"], "dists": {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}}]
    flagged = [{"images": ["a.png", "b.png", "c.png"], "dists": {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}, "qc": True}]
    _write_session(tmp_path, "v3a.csv", "A", good, version="3.0")
    _write_session(tmp_path, "v2a.csv", "B", good, version="2.0")
    _write_session(tmp_path, "v3bad.csv", "C", flagged, version="3.0")  # 100% flagged
    assert len(pilot.load_pilot_subjects(str(tmp_path), man)) == 3
    assert len(pilot.load_pilot_subjects(str(tmp_path), man, versions=["3.0"])) == 2
    assert len(pilot.load_pilot_subjects(str(tmp_path), man, apply_qc=True)) == 2  # drops the flagged one


def test_aggregate_raises_on_disconnected_graph(tmp_path):
    _, rel2idx = pilot.load_manifest(_write_manifest(tmp_path))
    # subject only ever touches images {a,b,c}; d,e never observed -> graph disconnected
    s = pilot.load_pilot_subject(
        _write_session(tmp_path, "s.csv", "P", [{"images": ["a.png", "b.png", "c.png"],
                                                 "dists": {(0, 1): 0.1, (0, 2): 0.4, (1, 2): 0.7}}]), rel2idx)
    with pytest.raises(RuntimeError, match="connected components"):
        pilot.pilot_aggregate([s])


# --------------------------------------------------------------------- simulation hook + calibration
def test_simulate_returns_per_subject_matching_nan_pattern():
    emb = build_ground_truth_embeddings(80, 4, seed=1)
    params = TaskV3ExperimentParameters(6, 8, 8, 0.4, 1, 0.25, 0.2)
    _, res, per_subject = simulate_task_v3_experiment(params, emb, np.random.default_rng(0),
                                                      verbose=False, return_per_subject=True)
    assert per_subject.shape == (6, res.distances.shape[0])
    # a per-subject entry is observed iff that subject contributed; agreement is a valid correlation
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


def test_calibrate_recovers_known_parameters():
    """Self-consistency: simulate a 'pilot' cohort at known (noise, dispersion), then recover them.

    Uses the simulated cohort's own per-subject vectors + test-retest as the targets (bypassing the
    CSV path), so this checks the fitting logic and the perspective-invariance of test-retest.
    """
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
