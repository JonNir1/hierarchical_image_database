import pandas as pd
import pytest

from analysis.prod.parser import load_prod_data
from prod_csv_helpers import write_demographics_csv, write_session_csv


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
        "shine_variant": "pre", "task_version": "3.06", "deployment_mode": "production",
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
        "shine_variant": "pre", "task_version": "3.06", "deployment_mode": "production",
        "pairwise_distances": '[{"src1":"c.png","src2":"d.png","distance":5}]',
        "final_locations": '[{"src":"c.png","x":100,"y":100}]',
        "catch_trial_target_location": "center",
        "centroid_x": "0.1", "centroid_y": "0.125", "cluster_mean_distance": "0.05",
    }
    row.update(overrides)
    return row


def _complete_session_rows(participant_id: str = "p1") -> list[dict]:
    """All 20 main trials + 4 catch trials for one participant."""
    rows = [_main_trial_row(participant_id=participant_id, trial_type=f"trial_{n}") for n in range(1, 21)]
    rows += [_catch_trial_row(participant_id=participant_id, trial_type=f"catch_{n}") for n in range(1, 5)]
    return rows


class TestLoadProdData:
    def test_happy_path_complete_session(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        write_session_csv(tmp_path, _complete_session_rows(), "session1.csv")

        data = load_prod_data(tmp_path)

        assert list(data.keys()) == ["trials", "status", "catch_trials"]
        assert len(data["trials"]) == 20
        assert len(data["catch_trials"]) == 4
        assert data["status"].iloc[0]["completion_status"] == "completed"
        assert data["trials"]["participant_id"].iloc[0] == "p1"

    def test_incomplete_session_excluded(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])
        # missing trial_20 -> incomplete
        rows = [r for r in _complete_session_rows() if r["trial_type"] != "trial_20"]
        write_session_csv(tmp_path, rows, "session1.csv")

        with pytest.warns(UserWarning, match="none complete"):
            data = load_prod_data(tmp_path)

        assert data["trials"].empty
        assert data["status"].iloc[0]["completion_status"] == "incomplete_session"

    def test_picks_complete_session_among_multiple_files(self, tmp_path):
        """A participant can have more than one session file (e.g. a reconnect after
        an abandoned attempt) -- the complete one must be used, regardless of
        which file is discovered first."""
        write_demographics_csv(tmp_path, [_demo_row()])
        incomplete_rows = [r for r in _complete_session_rows() if r["trial_type"] != "trial_5"]
        write_session_csv(tmp_path, incomplete_rows, "session_a_abandoned.csv")
        write_session_csv(tmp_path, _complete_session_rows(), "session_b_complete.csv")

        data = load_prod_data(tmp_path)

        assert data["status"].iloc[0]["completion_status"] == "completed"
        assert len(data["trials"]) == 20
        assert data["trials"]["session_file"].iloc[0] == "session_b_complete"

    def test_revoked_consent_excluded_from_trials(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row(**{"Status": "RETURNED"})])

        with pytest.warns(UserWarning, match="revoked consent"):
            data = load_prod_data(tmp_path)

        assert data["trials"].empty
        assert data["status"].iloc[0]["completion_status"] == "revoked_consent"

    def test_approved_with_no_session_file_excluded_from_trials(self, tmp_path):
        write_demographics_csv(tmp_path, [_demo_row()])

        with pytest.warns(UserWarning, match="no session file found"):
            data = load_prod_data(tmp_path)

        assert data["trials"].empty
        assert data["status"].iloc[0]["completion_status"] == "erroneous_completion"

    def test_missing_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prod_data(tmp_path / "does_not_exist")

    def test_no_demographics_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="demographic"):
            load_prod_data(tmp_path)
