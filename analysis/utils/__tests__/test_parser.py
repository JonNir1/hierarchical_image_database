import json
import math

import pandas as pd
import pytest

from analysis.utils.parser_v2 import (
    _count_moves,
    _trial_image_set,
    _normalise_locations,
    load_data,
    parse_pairwise_distances,
)
from pilot_csv_helpers import _SESSION_COLUMNS, write_demographics_csv, write_session_csv


def _demo_row(**overrides) -> dict:
    row = {
        "Submission id": "s1", "Participant id": "p1", "Status": "APPROVED",
        "Time taken": "1200", "Age": "25", "Sex": "Male",
        "Ethnicity simplified": "White", "Country of birth": "US",
        "Country of residence": "US", "Nationality": "American",
        "Language": "English", "Student status": "No", "Employment status": "Full-Time",
    }
    row.update(overrides)
    return row


def _pw(pairs: list[tuple[str, str, float]]) -> str:
    return json.dumps([{"src1": a, "src2": b, "distance": d} for a, b, d in pairs])


_IMAGES_A = [("a.png", "b.png", 0.1), ("a.png", "c.png", 0.5), ("b.png", "c.png", 0.9)]
_IMAGES_A_REPEAT = [("a.png", "b.png", 0.15), ("a.png", "c.png", 0.45), ("b.png", "c.png", 0.95)]
_IMAGES_B = [("d.png", "e.png", 0.2), ("d.png", "f.png", 0.6), ("e.png", "f.png", 0.8)]


def _trial_row(trial_type, pairs=_IMAGES_A, **overrides) -> dict:
    row = {
        "participant_id": "p1", "trial_type": trial_type,
        "sort_area_width": "1000", "sort_area_height": "800",
        "moves": '[{"x":1,"y":1}]', "rt": "5000", "qc_flag": "false",
        "shine_variant": "pre", "task_version": "4.0", "deployment_mode": "pilot",
        "pairwise_distances": _pw(pairs),
        "final_locations": '[{"src":"a.png","x":500,"y":400}]',
        "init_locations": "", "is_trial_repeat": "false", "repeat_of_trial_number": "",
        "block": "experimental",
    }
    row.update(overrides)
    return row


class TestLoadData:
    def test_practice_trials_excluded_and_trial_id_continuous(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [
            _trial_row("practice", block=""),
            _trial_row("practice_catch", block=""),
            _trial_row("trial_1", block="screening"),
            _trial_row("catch_1", pairs=_IMAGES_B, block="screening"),
            _trial_row("trial_2", block="experimental"),
        ], "session1.csv")

        data = load_data(tmp_path)
        df_t = data["trials"]

        assert len(df_t) == 3
        assert df_t["trial_id"].tolist() == [1, 2, 3]
        assert df_t["is_catch"].tolist() == [False, True, False]
        assert df_t["block_type"].tolist() == ["screening", "screening", "experimental"]

    def test_block_type_absent_column_defaults_experimental(self, tmp_path):
        """Pre-v4 files have no `block` column in the header at all (not just NaN)."""
        write_demographics_csv(tmp_path, [_demo_row()])
        cols = [c for c in _SESSION_COLUMNS if c != "block"]
        rows = [_trial_row("trial_1"), _trial_row("catch_1", pairs=_IMAGES_B)]
        df = pd.DataFrame([{c: row.get(c, "") for c in cols} for row in rows])
        df.to_csv(tmp_path / "session1.csv", index=False)

        data = load_data(tmp_path)
        df_t = data["trials"]
        assert (df_t["block_type"] == "experimental").all()

    def test_repeat_of_trial_resolves_and_reliability_computed(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [
            _trial_row("trial_1"),
            _trial_row("trial_2", pairs=_IMAGES_A_REPEAT, is_trial_repeat="true", repeat_of_trial_number="1"),
        ], "session1.csv")

        data = load_data(tmp_path)
        df_t = data["trials"]

        orig = df_t[df_t["trial_id"] == 1].iloc[0]
        repeat = df_t[df_t["trial_id"] == 2].iloc[0]
        assert pd.isna(orig["repeat_of_trial"])
        assert repeat["repeat_of_trial"] == 1
        assert repeat["reliability"] == pytest.approx(1.0)  # same rank order -> r=1

    def test_repeat_image_set_mismatch_detected(self, tmp_path):
        from analysis.utils.parser_v2 import _validate_trial_repeat_image_sets_v2

        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [
            _trial_row("trial_1", pairs=_IMAGES_A),
            _trial_row("trial_2", pairs=_IMAGES_B, is_trial_repeat="true", repeat_of_trial_number="1"),
        ], "session1.csv")

        with pytest.warns(UserWarning, match="do not share the same image set"):
            data = load_data(tmp_path)

        report = _validate_trial_repeat_image_sets_v2(data["trials"])
        assert not report.iloc[0]["images_match"]

    def test_reliability_none_when_too_few_shared_pairs(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [
            _trial_row("trial_1", pairs=[("a.png", "b.png", 0.1)]),
            _trial_row("trial_2", pairs=[("a.png", "b.png", 0.2)],
                       is_trial_repeat="true", repeat_of_trial_number="1"),
        ], "session1.csv")

        data = load_data(tmp_path)
        repeat = data["trials"][data["trials"]["trial_id"] == 2].iloc[0]
        assert pd.isna(repeat["reliability"])

    def test_malformed_pairwise_distances_warns_and_nulls_reliability(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [
            _trial_row("trial_1", pairwise_distances='[{"src1":"a.png","src2":"b.png","distance":0.1}'),  # truncated
            _trial_row("trial_2", is_trial_repeat="true", repeat_of_trial_number="1"),
        ], "session1.csv")

        with pytest.warns(UserWarning, match="failed to parse|malformed"):
            data = load_data(tmp_path)

        repeat = data["trials"][data["trials"]["trial_id"] == 2].iloc[0]
        assert pd.isna(repeat["reliability"])

    def test_zero_real_trial_rows_is_missing_data(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [
            {**_trial_row("pavlovia"), "trial_type": "pavlovia"},
        ], "session1.csv")

        data = load_data(tmp_path)
        assert data["participants"].iloc[0]["status"] == "missing data"
        assert data["trials"].empty

    def test_all_four_status_values(self, tmp_path):
        write_demographics_csv(tmp_path, [
            _demo_row(**{"Submission id": "s1", "Participant id": "p1"}),  # full data
            _demo_row(**{"Submission id": "s2", "Participant id": "p2", "Status": "RETURNED"}),  # revoked
            _demo_row(**{"Submission id": "s3", "Participant id": "p3"}),  # missing data (no file)
            _demo_row(**{"Submission id": "s4", "Participant id": "p4"}),  # screened out
        ])
        write_session_csv(tmp_path, [_trial_row("trial_1", **{"participant_id": "p1"})], "session_p1.csv")
        write_session_csv(tmp_path, [
            {**_trial_row("trial_1", **{"participant_id": "p4"}), "trial_type": "trial_1"},
            {
                "participant_id": "p4", "trial_type": "screening_eval", "task_version": "4.0",
                "deployment_mode": "pilot", "pass": "false", "reasons": "[]",
                "move_ratio_fail_rate": "0.5", "distance_sd_fail_rate": "0.0",
                "min_reliability": "-0.1", "median_reliability": "0.1",
                "sort_area_width": "1000", "sort_area_height": "800", "shine_variant": "pre",
            },
        ], "session_p4.csv")

        data = load_data(tmp_path)
        status_by_pid = data["participants"].set_index("participant_id")["status"]
        assert status_by_pid["p1"] == "full data"
        assert status_by_pid["p2"] == "revoked consent"
        assert status_by_pid["p3"] == "missing data"
        assert status_by_pid["p4"] == "screened out"

    def test_screening_eval_diagnostics_preserved_and_absent_when_no_row(self, tmp_path):
        write_demographics_csv(tmp_path, [
            _demo_row(**{"Submission id": "s1", "Participant id": "p1"}),  # screened out, has diagnostics
            _demo_row(**{"Submission id": "s2", "Participant id": "p2"}),  # full data, pre-v4, no screening_eval row
        ])
        write_session_csv(tmp_path, [
            {**_trial_row("trial_1", **{"participant_id": "p1"}), "trial_type": "trial_1"},
            {
                "participant_id": "p1", "trial_type": "screening_eval", "task_version": "4.0",
                "deployment_mode": "pilot", "pass": "false", "reasons": '["move-ratio fail rate 0.5"]',
                "move_ratio_fail_rate": "0.5", "distance_sd_fail_rate": "0.0",
                "min_reliability": "-0.1", "median_reliability": "0.1",
                "sort_area_width": "1000", "sort_area_height": "800", "shine_variant": "pre",
            },
        ], "session_p1.csv")
        write_session_csv(tmp_path, [_trial_row("trial_1", **{"participant_id": "p2"}, task_version="1.0")],
                           "session_p2.csv")

        data = load_data(tmp_path)
        by_pid = data["participants"].set_index("participant_id")

        assert by_pid.loc["p1", "reasons"] == '["move-ratio fail rate 0.5"]'
        assert by_pid.loc["p1", "move_ratio_fail_rate"] == pytest.approx(0.5)
        assert by_pid.loc["p1", "distance_sd_fail_rate"] == pytest.approx(0.0)
        assert by_pid.loc["p1", "min_reliability"] == pytest.approx(-0.1)
        assert by_pid.loc["p1", "median_reliability"] == pytest.approx(0.1)

        assert pd.isna(by_pid.loc["p2", "min_reliability"])
        assert pd.isna(by_pid.loc["p2", "median_reliability"])
        assert pd.isna(by_pid.loc["p2", "reasons"])

    def test_cohort_raw_passthrough_and_prefix_fallback(self, tmp_path):
        write_demographics_csv(tmp_path, [
            _demo_row(**{"Submission id": "s1", "Participant id": "p1"}),
            _demo_row(**{"Submission id": "s2", "Participant id": "p2"}),
        ])
        write_session_csv(tmp_path, [_trial_row("trial_1", **{"participant_id": "p1"},
                                                 deployment_mode="production")], "session_p1.csv")
        write_session_csv(tmp_path, [_trial_row("trial_1", **{"participant_id": "p2"},
                                                 deployment_mode="")], "prod_session_p2.csv")

        data = load_data(tmp_path)
        cohort_by_pid = data["participants"].set_index("participant_id")["cohort"]
        assert cohort_by_pid["p1"] == "production"
        assert cohort_by_pid["p2"] == "production"  # from prod_ filename prefix fallback

    def test_num_associated_files(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [_trial_row("trial_1")], "session_a.csv")
        write_session_csv(tmp_path, [_trial_row("trial_1"), _trial_row("trial_2")], "session_b.csv")

        data = load_data(tmp_path)
        row = data["participants"].iloc[0]
        assert row["num_associated_files"] == 2
        assert row["file_name"] == "session_b.csv"  # more real trial rows wins

    def test_multi_file_recency_tiebreak(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [_trial_row("trial_1")],
                           "hierarchical-image-database_PARTICIPANT_SESSION_2026-07-24_20h11.13.978.csv")
        write_session_csv(tmp_path, [_trial_row("trial_1", pairs=_IMAGES_B)],
                           "hierarchical-image-database_PARTICIPANT_SESSION_2026-07-24_20h55.35.371.csv")

        data = load_data(tmp_path)
        row = data["participants"].iloc[0]
        assert "20h55.35.371" in row["file_name"]  # most recent wins the tie

    def test_multi_file_tied_timestamp_raises(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [_trial_row("trial_1")],
                           "hierarchical-image-database_PARTICIPANT_SESSION_2026-07-24_20h11.13.978.csv")
        write_session_csv(tmp_path, [_trial_row("trial_1", pairs=_IMAGES_B)],
                           "hierarchical-image-database_PARTICIPANT_SESSION2_2026-07-24_20h11.13.978.csv")

        with pytest.raises(ValueError, match="cannot resolve"):
            load_data(tmp_path)

    def test_multi_file_unparseable_timestamp_raises(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [_trial_row("trial_1")], "session_no_timestamp_a.csv")
        write_session_csv(tmp_path, [_trial_row("trial_1", pairs=_IMAGES_B)], "session_no_timestamp_b.csv")

        with pytest.raises(ValueError, match="Cannot parse a timestamp"):
            load_data(tmp_path)

    @pytest.mark.parametrize("field,bad_value", [
        ("deployment_mode", "production"),
        ("shine_variant", "post"),
        ("sort_area_width", "500"),
        ("sort_area_height", "400"),
        ("task_version", "3.0"),
    ])
    def test_constancy_check_raises_on_inconsistent_field(self, tmp_path, field, bad_value):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [
            _trial_row("trial_1"),
            _trial_row("trial_2", **{field: bad_value}),
        ], "session1.csv")

        with pytest.raises(ValueError, match=field):
            load_data(tmp_path)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Shared trial/demographics helpers
#
# Moved here with the helpers themselves when they migrated out of analysis.utils.parser,
# so their coverage survives that module's retirement. Behaviour is unchanged; only the
# import location moved.
# ---------------------------------------------------------------------------

class TestNormaliseLocations:
    def test_normalises_pixel_coordinates(self):
        result = _normalise_locations('[{"x":100,"y":200}]', canvas_w=1000, canvas_h=800)
        assert json.loads(result) == [{"x": 0.1, "y": 0.25}]

    def test_empty_string_passthrough(self):
        assert _normalise_locations("", canvas_w=1000, canvas_h=800) == ""

    def test_nan_passthrough(self):
        result = _normalise_locations(float("nan"), canvas_w=1000, canvas_h=800)
        assert math.isnan(result)

    def test_malformed_json_passthrough(self):
        malformed = '{"not": "a list"'
        assert _normalise_locations(malformed, canvas_w=1000, canvas_h=800) == malformed


class TestCountMoves:
    def test_counts_entries(self):
        assert _count_moves('[{"x":1,"y":1},{"x":2,"y":2}]') == 2

    def test_empty_string_is_zero(self):
        assert _count_moves("") == 0

    def test_nan_is_zero(self):
        assert _count_moves(float("nan")) == 0

    def test_malformed_json_is_zero(self):
        assert _count_moves("not json") == 0


class TestParsePairwiseDistances:
    def test_parses_and_sorts_pair_keys(self):
        result = parse_pairwise_distances('[{"src1":"b.png","src2":"a.png","distance":3.5}]')
        assert result == {("a.png", "b.png"): 3.5}

    def test_empty_string_is_empty_dict(self):
        assert parse_pairwise_distances("") == {}

    def test_nan_is_empty_dict(self):
        assert parse_pairwise_distances(float("nan")) == {}

    def test_malformed_json_is_empty_dict(self):
        assert parse_pairwise_distances("not json") == {}

    def test_trial_image_set_is_union_of_all_pairs(self):
        pw_json = '[{"src1":"a.png","src2":"b.png","distance":1},{"src1":"b.png","src2":"c.png","distance":2}]'
        assert _trial_image_set(pw_json) == frozenset({"a.png", "b.png", "c.png"})
