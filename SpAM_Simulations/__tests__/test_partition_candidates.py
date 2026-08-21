"""Tests for the shared retained / early-fail / false-alarm partition.

One implementation, because two of them drift. The v6 calibration gate and the v6 ground-truth
rebuild both need this split, and an earlier copy that checked only reliability on the experimental
block - not move-ratio - counted 73 retained where every other analysis counted 67.
"""
import json

import numpy as np
import pandas as pd
import pytest

from SpAM_Simulations.empirical import screening_audit as sa

THRESHOLDS = {
    "min_move_item_ratio": 1.0,
    "min_pairwise_distance_sd": 0.01,
    "move_ratio_max_fail_rate": 0.13,
    "distance_sd_max_fail_rate": 0.13,
    "min_reliability": 0.0,
    "median_reliability": None,
}


def _participants(rows):
    return pd.DataFrame(rows)


def _trials(pid, n=14, *, moves=40, sd=0.3, reliabilities=(0.5, 0.5), block="experimental"):
    """A parser-style experimental block for one subject, with `reliabilities` on repeat trials."""
    rows = []
    for i in range(n):
        # The parser's own shape: a JSON list of {src1, src2, distance}. A dict of "a|b" keys parses
        # to zero items and zero spread, which silently fails every subject on distance-SD.
        pairwise = json.dumps([
            {"src1": "a", "src2": "b", "distance": sd},
            {"src1": "a", "src2": "c", "distance": 0.0},
            {"src1": "b", "src2": "c", "distance": 2 * sd},
        ])
        rows.append({
            "participant_id": pid, "is_catch": False, "block_type": block,
            "num_moves": moves, "pairwise_distances": pairwise,
            "reliability": reliabilities[i] if i < len(reliabilities) else np.nan,
        })
    return rows


@pytest.fixture
def frames():
    people = [
        # clean on both blocks
        {"participant_id": "keep", "cohort": "production", "status": "full data",
         "min_reliability": 0.4, "move_ratio_fail_rate": 0.0, "distance_sd_fail_rate": 0.0},
        # failed the gate in-task
        {"participant_id": "early", "cohort": "production", "status": "screened out",
         "min_reliability": -0.2, "move_ratio_fail_rate": 0.0, "distance_sd_fail_rate": 0.0},
        # cleared the gate, then failed the experimental block on RELIABILITY
        {"participant_id": "fa_rel", "cohort": "production", "status": "full data",
         "min_reliability": 0.4, "move_ratio_fail_rate": 0.0, "distance_sd_fail_rate": 0.0},
        # cleared the gate, then failed the experimental block on MOVE-RATIO only
        {"participant_id": "fa_move", "cohort": "production", "status": "full data",
         "min_reliability": 0.4, "move_ratio_fail_rate": 0.0, "distance_sd_fail_rate": 0.0},
    ]
    trials = (_trials("keep") + _trials("fa_rel", reliabilities=(0.5, -0.3))
              + _trials("fa_move", moves=0))
    return _participants(people), pd.DataFrame(trials)


class TestPartition:
    def test_it_sorts_everyone_into_exactly_one_group(self, frames):
        part = sa.partition_candidates(*frames, THRESHOLDS)
        assert part["retained"] == ["keep"]
        assert part["early_fail"] == ["early"]
        assert sorted(part["false_alarm"]) == ["fa_move", "fa_rel"]

    def test_move_ratio_failures_in_the_experimental_block_are_false_alarms(self, frames):
        """The 73-vs-67 bug. Reliability alone would call `fa_move` retained."""
        part = sa.partition_candidates(*frames, THRESHOLDS)
        assert "fa_move" not in part["retained"]
        assert "fa_move" in part["false_alarm"]

    def test_every_candidate_is_accounted_for(self, frames):
        part = sa.partition_candidates(*frames, THRESHOLDS)
        assert sum(len(v) for v in part.values()) == 4

    def test_a_stricter_threshold_moves_people_out_of_retained(self, frames):
        loose = sa.partition_candidates(*frames, THRESHOLDS, threshold=0.0)
        strict = sa.partition_candidates(*frames, THRESHOLDS, threshold=0.45)
        assert loose["retained"] == ["keep"]
        assert strict["retained"] == []
        assert "keep" in strict["early_fail"]

    def test_non_attempting_statuses_are_excluded(self, frames):
        """A revoked consent never produced a gate decision; counting it as a clean pass inflates
        the pass rate, which is a bug this codebase has already had once."""
        participants, trials = frames
        participants = pd.concat([participants, _participants([
            {"participant_id": "ghost", "cohort": "production", "status": "revoked consent",
             "min_reliability": np.nan, "move_ratio_fail_rate": np.nan,
             "distance_sd_fail_rate": np.nan}])], ignore_index=True)
        part = sa.partition_candidates(participants, trials, THRESHOLDS)
        assert "ghost" not in sum(part.values(), [])

    def test_other_cohorts_are_excluded(self, frames):
        participants, trials = frames
        participants = participants.assign(cohort="pilot")
        part = sa.partition_candidates(participants, trials, THRESHOLDS)
        assert sum(len(v) for v in part.values()) == 0

    def test_a_passer_with_no_experimental_block_is_an_early_fail(self, frames):
        """Screened out mid-task at the deployed gate, but clearing a LOWER counterfactual one.
        They contributed no experimental data, so they cannot be retained."""
        participants, trials = frames
        part = sa.partition_candidates(participants, trials, THRESHOLDS, threshold=-1.0)
        assert "early" in part["early_fail"]
        assert "early" not in part["retained"]

    def test_a_null_threshold_disables_its_criterion(self, frames):
        """Mirrors the JS: a null threshold means the criterion is off, not that it always fails."""
        participants, trials = frames
        thresholds = {**THRESHOLDS, "move_ratio_max_fail_rate": None}
        part = sa.partition_candidates(participants, trials, thresholds)
        assert "fa_move" in part["retained"]
