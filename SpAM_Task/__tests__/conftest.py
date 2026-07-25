import sys
from pathlib import Path
import pytest
from unittest.mock import patch

# Make SpAM_Task/ importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_manifest as gm


@pytest.fixture
def task_env(tmp_path):
    """Patch module-level path constants to a temp directory, yield it.

    Both TASK_DIR and PROJECT_ROOT are patched to tmp_path. In production
    TASK_DIR is SpAM_Task/ and PROJECT_ROOT is its parent, but the tests
    treat tmp_path as both the script's home AND the project root so that
    relative paths in test configs resolve into tmp_path.
    """
    with (
        patch.object(gm, "TASK_DIR", tmp_path),
        patch.object(gm, "PROJECT_ROOT", tmp_path),
        patch.object(gm, "CONFIG_PATH", tmp_path / "task_config.json"),
        patch.object(gm, "MANIFEST_PATH", tmp_path / "stimuli_manifest.json"),
    ):
        yield tmp_path


def make_config(
    base: Path,
    *,
    main_root: str = "",
    stimuli_path: str = "",
    images_per_trial: int = 20,
    **extra,
) -> dict:
    """Write a minimal task_config.json into *base*.

    Returns the config dict. `main_root` is the value of
    `experimental_trials.stimuli_path` (main-image root; generate_manifest scans
    both `<main_root>/pre_shine/` and `<main_root>/post_shine/`). `stimuli_path`
    is `catch_trials.stimuli_path`, shared by both the practice and catch
    secondary sets; `images_per_trial` is `catch_trials.images_per_trial`, which
    is the threshold for BOTH secondary sets (the practice trial has no count of
    its own -- see generate_manifest.main).
    """
    config = {
        "experimental_trials": {
            "stimuli_path": main_root,
        },
        "catch_trials": {
            "stimuli_path":     stimuli_path,
            "images_per_trial": images_per_trial,
        },
        **extra,
    }
    (base / "task_config.json").write_text(__import__("json").dumps(config))
    return config
