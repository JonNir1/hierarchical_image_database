import json
import math

import pandas as pd
import pytest

from analysis.utils.parser import (
    _count_moves,
    _load_demographics,
    _normalise_locations,
    _trial_image_set,
    load_pilot_data,
    parse_pairwise_distances,
    validate_trial_repeat_image_sets,
)
from conftest import write_demographics_csv, write_session_csv


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


def _main_trial_row(**overrides) -> dict:
    row = {
        "participant_id": "p1", "trial_type": "trial_1",
        "sort_area_width": "1000", "sort_area_height": "800",
        "moves": '[{"x":1,"y":1}]', "rt": "5000", "qc_flag": "false",
        "shine_variant": "pre", "task_version": "1.0", "deployment_mode": "pilot",
        "pairwise_distances": '[{"src1":"a.png","src2":"b.png","distance":10}]',
        "final_locations": '[{"src":"a.png","x":500,"y":400}]',
        "init_locations": "", "is_trial_repeat": "false", "repeat_of_trial_number": "",
    }
    row.update(overrides)
    return row


def _catch_trial_row(**overrides) -> dict:
    row = {
        "participant_id": "p1", "trial_type": "catch_1",
        "sort_area_width": "1000", "sort_area_height": "800",
        "moves": "[]", "rt": "3000", "qc_flag": "true",
        "shine_variant": "pre", "task_version": "1.0", "deployment_mode": "pilot",
        "pairwise_distances": '[{"src1":"c.png","src2":"d.png","distance":5}]',
        "final_locations": '[{"src":"c.png","x":100,"y":100}]',
        "catch_trial_target_location": "center",
        "centroid_x": "0.1", "centroid_y": "0.125", "cluster_mean_distance": "0.05",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# load_pilot_data
# ---------------------------------------------------------------------------

class TestLoadPilotData:
    def test_happy_path(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, [_main_trial_row(), _catch_trial_row()], "session1.csv")

        data = load_pilot_data(tmp_path)

        assert list(data.keys()) == ["trials", "status", "catch_trials"]

        df_trials = data["trials"]
        assert len(df_trials) == 1
        row = df_trials.iloc[0]
        assert row["trial_number"] == 1
        assert row["n_moves"] == 1
        assert bool(row["qc_flag"]) is False
        assert row["participant_id"] == "p1"
        assert row["age"] == 25
        final_locations = json.loads(row["final_locations"])
        assert final_locations == [{"src": "a.png", "x": 0.5, "y": 0.5}]

        df_catch = data["catch_trials"]
        assert len(df_catch) == 1
        crow = df_catch.iloc[0]
        assert crow["catch_number"] == 1
        assert crow["n_moves"] == 0
        assert bool(crow["qc_flag"]) is True
        assert crow["catch_trial_target_location"] == "center"
        assert crow["sort_area_width"] == 1000
        assert crow["sort_area_height"] == 800

        df_status = data["status"]
        assert len(df_status) == 1
        assert df_status.iloc[0]["completion_status"] == "completed"

    def test_missing_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pilot_data(tmp_path / "does_not_exist")

    def test_path_is_file_raises_not_a_directory(self, tmp_path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        with pytest.raises(NotADirectoryError):
            load_pilot_data(file_path)

    def test_no_demographics_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="demographics"):
            load_pilot_data(tmp_path)

    def test_duplicate_participant_id_warns(self, tmp_path):
        write_demographics_csv(tmp_path, [
            _demo_row(**{"Submission id": "s1"}),
            _demo_row(**{"Submission id": "s2"}),
        ])
        write_session_csv(tmp_path, [_main_trial_row()], "session1.csv")

        with pytest.warns(UserWarning, match="Duplicate participant_id"):
            load_pilot_data(tmp_path)

    def test_revoked_consent_excluded_from_trials(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row(**{"Status": "RETURNED"})])

        with pytest.warns(UserWarning, match="revoked consent"):
            data = load_pilot_data(tmp_path)

        assert data["trials"].empty
        assert data["status"].iloc[0]["completion_status"] == "revoked_consent"

    def test_approved_with_no_session_file_excluded_from_trials(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])

        with pytest.warns(UserWarning, match="no session file found"):
            data = load_pilot_data(tmp_path)

        assert data["trials"].empty
        assert data["status"].iloc[0]["completion_status"] == "erroneous_completion"


# ---------------------------------------------------------------------------
# _load_demographics
# ---------------------------------------------------------------------------

class TestLoadDemographics:
    def test_renames_columns(self, tmp_path):
        path = write_demographics_csv(tmp_path, [_demo_row()])
        df = _load_demographics(path)
        assert "participant_id" in df.columns
        assert "prolific_status" in df.columns
        assert "Participant id" not in df.columns

    def test_masks_prolific_sentinels(self, tmp_path):
        path = write_demographics_csv(tmp_path, [_demo_row(**{"Ethnicity simplified": "DATA_EXPIRED"})])
        df = _load_demographics(path)
        assert pd.isna(df.iloc[0]["ethnicity"])

    def test_coerces_age_to_numeric(self, tmp_path):
        path = write_demographics_csv(tmp_path, [_demo_row(**{"Age": "not-a-number"})])
        df = _load_demographics(path)
        assert pd.isna(df.iloc[0]["age"])


# ---------------------------------------------------------------------------
# _normalise_locations
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


# ---------------------------------------------------------------------------
# _count_moves
# ---------------------------------------------------------------------------

class TestCountMoves:
    def test_counts_entries(self):
        assert _count_moves('[{"x":1,"y":1},{"x":2,"y":2}]') == 2

    def test_empty_string_is_zero(self):
        assert _count_moves("") == 0

    def test_nan_is_zero(self):
        assert _count_moves(float("nan")) == 0

    def test_malformed_json_is_zero(self):
        assert _count_moves("not json") == 0


# ---------------------------------------------------------------------------
# parse_pairwise_distances / _trial_image_set
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# validate_trial_repeat_image_sets
# ---------------------------------------------------------------------------

class TestValidateTrialRepeatImageSets:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "participant_id": "p1", "session_file": "session1", "trial_number": 1,
                "is_trial_repeat": False, "repeat_of_trial_number": pd.NA,
                "pairwise_distances": '[{"src1":"a.png","src2":"b.png","distance":1}]',
            },
            {
                "participant_id": "p1", "session_file": "session1", "trial_number": 2,
                "is_trial_repeat": True, "repeat_of_trial_number": 1,
                "pairwise_distances": '[{"src1":"a.png","src2":"b.png","distance":1}]',
            },
            {
                "participant_id": "p1", "session_file": "session1", "trial_number": 3,
                "is_trial_repeat": False, "repeat_of_trial_number": pd.NA,
                "pairwise_distances": '[{"src1":"c.png","src2":"d.png","distance":1}]',
            },
            {
                "participant_id": "p1", "session_file": "session1", "trial_number": 4,
                "is_trial_repeat": True, "repeat_of_trial_number": 3,
                # deliberately different images than trial 3
                "pairwise_distances": '[{"src1":"e.png","src2":"f.png","distance":1}]',
            },
        ])

    def test_matching_image_sets_pass(self):
        report = validate_trial_repeat_image_sets(self._base_df())
        row = report[report["trial_number"] == 2].iloc[0]
        assert bool(row["images_match"]) is True

    def test_mismatched_image_sets_fail(self):
        report = validate_trial_repeat_image_sets(self._base_df())
        row = report[report["trial_number"] == 4].iloc[0]
        assert bool(row["images_match"]) is False

    def test_missing_required_column_raises_key_error(self):
        df = self._base_df().drop(columns=["pairwise_distances"])
        with pytest.raises(KeyError):
            validate_trial_repeat_image_sets(df)
