"""Tests for analysis.behavioral.demographics_and_behavior's participant filter."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.behavioral.demographics_and_behavior import retained_participants


def test_retained_participants_applies_every_exclusion_and_rejects_unknown_cohort():
    df_participants = pd.DataFrame([
        {"participant_id": "kept", "cohort": "production", "status": "full data", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "kept_later_version", "cohort": "production", "status": "full data", "shine_variant": "pre", "task_version": 4.01},
        {"participant_id": "pilot", "cohort": "pilot", "status": "full data", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "screened_out", "cohort": "production", "status": "screened out", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "false_positive", "cohort": "production", "status": "false_positive", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "old_version", "cohort": "production", "status": "full data", "shine_variant": "pre", "task_version": 3.0},
        {"participant_id": "other_variant", "cohort": "production", "status": "full data", "shine_variant": "post", "task_version": 4.0},
    ])

    pre = retained_participants(df_participants, "pre")
    assert pre["participant_id"].tolist() == ["kept", "kept_later_version"]
    assert retained_participants(df_participants, "post")["participant_id"].tolist() == ["other_variant"]

    with pytest.raises(ValueError, match="cohort must be one of"):
        retained_participants(df_participants, "mid")
