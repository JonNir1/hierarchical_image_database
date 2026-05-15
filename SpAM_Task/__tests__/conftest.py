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
        patch.object(gm, "CONFIG_PATH", tmp_path / "task_config.json"),
        patch.object(gm, "MANIFEST_PATH", tmp_path / "stimuli_manifest.json"),
    ):
        yield tmp_path


def make_config(
    base: Path,
    *,
    main_root: str = "",
    stimuli_practice_path: str = "",
    stimuli_catch_path: str = "",
    shine_variant: str = "pre",
    images_per_trial: int = 20,
    practice_images_per_trial: int = 8,
    **extra,
) -> dict:
    """Write a minimal task_config.json (new nested schema) into *base*.

    Returns the config dict. `main_root` is the value of
    `stimuli_paths.main_root`; the main directory resolves to
    `<main_root>/<shine_variant>_shine/` per `generate_manifest.resolve_main_dir`.
    """
    config = {
        "stimuli_paths": {
            "main_root": main_root,
            "practice":  stimuli_practice_path,
            "catch":     stimuli_catch_path,
        },
        "shine": {
            "shine_variant": shine_variant,
        },
        "design": {
            "practice_images_per_trial": practice_images_per_trial,
        },
        "catch_trials": {
            "images_per_trial": images_per_trial,
        },
        **extra,
    }
    (base / "task_config.json").write_text(__import__("json").dumps(config))
    return config
