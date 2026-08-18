"""Tests for the production-tracking tables.

Two of these pin bugs that were live and silent: a provenance stamp that overwrote a real column,
and a pass rate computed over people who never sat the task. Both produced plausible-looking
numbers, which is exactly why they are worth a test.
"""
import numpy as np
import pandas as pd
import pytest

from SpAM_Simulations.empirical import prod_tracking as pt


class _FakeSubject:
    def __init__(self, pid, variant, n_pairs=6, seed=0):
        rng = np.random.default_rng(seed)
        self.participant_id = pid
        self.shine_variant = variant
        self.task_version = 4.0
        self.distances = rng.random(n_pairs)
        self.n_obs = np.ones(n_pairs, dtype=int)
        a = rng.random(n_pairs)
        self.retest_pairs = [(a, a + rng.normal(0, 0.05, n_pairs))]

    def num_observed_pairs(self):
        return int((self.n_obs > 0).sum())


def _cohort(n_pre=3, n_post=3):
    subs = [_FakeSubject(f"PRE_{i}", "pre", seed=i) for i in range(n_pre)]
    subs += [_FakeSubject(f"POST_{i}", "post", seed=100 + i) for i in range(n_post)]
    return subs


class TestTheStampDoesNotClobberRealColumns:
    """`_with_stamp` uses `.assign`, so an unprefixed key silently overwrites a same-named column.

    It did: every per-cohort table reported the pooled subject count as its own `n_subjects`, which
    read as a plausible number and was wrong.
    """

    def test_stamp_keys_are_prefixed(self):
        stamp = pt._stamp(_cohort())
        assert "snapshot_n_subjects" in stamp
        assert "n_subjects" not in stamp, "an unprefixed key would overwrite per-cohort columns"

    def test_a_per_group_n_subjects_column_survives_stamping(self):
        subs = _cohort(n_pre=3, n_post=5)
        frame = pd.DataFrame([{"group": "pre", "n_subjects": 3},
                              {"group": "post", "n_subjects": 5}])
        stamped = pt._with_stamp(frame, subs)
        assert list(stamped["n_subjects"]) == [3, 5]
        assert set(stamped["snapshot_n_subjects"]) == {8}


class TestScreeningOutcomesCountOnlyPeopleWhoSatTheTask:
    """A pass rate over non-participants is not a pass rate.

    "revoked consent" and "missing data" are recruitment attrition. Counting them as clean passes
    inflated both the denominator and the numerator, and the resulting rate looked reasonable.
    """

    def _participants(self):
        return pd.DataFrame([
            {"participant_id": "A", "cohort": "production", "status": "full data",
             "shine_variant": "pre", "reasons": None},
            {"participant_id": "B", "cohort": "production", "status": "screened out",
             "shine_variant": "pre", "reasons": '["minimum reliability -0.2 is below x"]'},
            {"participant_id": "C", "cohort": "production", "status": "revoked consent",
             "shine_variant": "post", "reasons": None},
            {"participant_id": "D", "cohort": "pilot", "status": "full data",
             "shine_variant": "pre", "reasons": None},
        ])

    def test_non_attempters_are_excluded_and_counted_separately(self, monkeypatch):
        participants = self._participants()
        trials = pd.DataFrame(columns=["participant_id", "is_catch", "block_type",
                                       "pairwise_distances", "num_moves", "reliability"])
        monkeypatch.setattr("analysis.utils.parser.load_data",
                            lambda d: {"participants": participants, "trials": trials})
        out = pt.screening_outcomes("ignored", "ignored")
        assert sorted(out["participant_id"]) == ["A", "B"], "C revoked consent, D is pilot"
        assert set(out["n_not_attempted"]) == {1}

    def test_the_pilot_cohort_is_never_included(self, monkeypatch):
        participants = self._participants()
        trials = pd.DataFrame(columns=["participant_id", "is_catch", "block_type",
                                       "pairwise_distances", "num_moves", "reliability"])
        monkeypatch.setattr("analysis.utils.parser.load_data",
                            lambda d: {"participants": participants, "trials": trials})
        out = pt.screening_outcomes("ignored", "ignored")
        assert "D" not in set(out["participant_id"])

    def test_summary_rates_use_the_attempted_denominator(self):
        outcomes = pd.DataFrame([
            {"participant_id": "A", "shine_variant": "pre", "outcome": "clean pass",
             "failed_reliability": False, "failed_move_ratio": False, "failed_distance_sd": False},
            {"participant_id": "B", "shine_variant": "pre", "outcome": "early fail",
             "failed_reliability": True, "failed_move_ratio": False, "failed_distance_sd": False},
            {"participant_id": "C", "shine_variant": "post", "outcome": "false positive",
             "failed_reliability": False, "failed_move_ratio": True, "failed_distance_sd": False},
        ])
        summary = pt.screening_summary(outcomes)
        pooled = summary[summary["group"] == pt.POOLED].iloc[0]
        assert pooled["n_candidates"] == 3
        assert pooled["early_fail"] == 1
        assert pooled["pass_rate"] == pytest.approx(2 / 3)
        # Two of three cleared the gate; one of those two then failed on the experimental block.
        assert pooled["false_positive_rate"] == pytest.approx(1 / 2)


class TestGroupingFollowsThePlan:
    def test_participant_behaviour_is_pooled_first_then_split(self):
        groups = [label for label, _ in pt.by_variant(_cohort())]
        assert groups == [pt.POOLED, "pre", "post"]

    def test_a_missing_variant_is_simply_absent(self):
        groups = [label for label, _ in pt.by_variant(_cohort(n_pre=3, n_post=0))]
        assert groups == [pt.POOLED, "pre"]

    def test_reliability_keeps_one_row_per_subject_for_the_decile_rule(self):
        """The pre-registration excludes the lowest decile per cohort, which needs a distribution."""
        frame = pt.reliability(_cohort(n_pre=3, n_post=3))
        assert len(frame) == 6
        assert set(frame["shine_variant"]) == {"pre", "post"}
