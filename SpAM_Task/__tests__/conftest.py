import sys
from pathlib import Path
import pytest
from unittest.mock import patch

# Make SpAM_Task/ importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_manifest as gm


@pytest.fixture
def task_env(tmp_path):
    """Patch module-level path constants to a temp directory, yield it."""
    with (
        patch.object(gm, "TASK_DIR", tmp_path),
        patch.object(gm, "CONFIG_PATH", tmp_path / "config.json"),
        patch.object(gm, "MANIFEST_PATH", tmp_path / "stimuli_manifest.json"),
    ):
        yield tmp_path


def make_config(base: Path, **overrides) -> dict:
    """Write a minimal config.json into *base* and return the dict."""
    config = {
        "stimuli_path": "",
        "stimuli_practice_path": "",
        "stimuli_catch_path": "",
        "images_per_trial": 20,
        "practice_images_per_trial": 8,
        **overrides,
    }
    (base / "config.json").write_text(__import__("json").dumps(config))
    return config
