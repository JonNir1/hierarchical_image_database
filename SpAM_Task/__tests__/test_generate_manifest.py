import json
from pathlib import Path
from unittest.mock import patch

import pytest

import generate_manifest as gm
from conftest import make_config


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_absolute_path_returned_unchanged(self, tmp_path):
        target = tmp_path / "some" / "dir"
        with patch.object(gm, "PROJECT_ROOT", tmp_path):
            assert gm.resolve_path(str(target)) == target

    def test_relative_path_resolved_against_project_root(self, tmp_path):
        with patch.object(gm, "PROJECT_ROOT", tmp_path):
            result = gm.resolve_path("stimuli")
        assert result == (tmp_path / "stimuli").resolve()

    def test_relative_path_with_subdirs(self, tmp_path):
        with patch.object(gm, "PROJECT_ROOT", tmp_path):
            result = gm.resolve_path("a/b/c")
        assert result == (tmp_path / "a" / "b" / "c").resolve()


# ---------------------------------------------------------------------------
# scan_pngs
# ---------------------------------------------------------------------------

class TestScanPngs:
    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert gm.scan_pngs(tmp_path) == []

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / "a.PNG").touch()
        (tmp_path / "b.Png").touch()
        (tmp_path / "c.png").touch()
        assert len(gm.scan_pngs(tmp_path)) == 3

    def test_ignores_non_png_files(self, tmp_path):
        (tmp_path / "image.jpg").touch()
        (tmp_path / "image.gif").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "image.png").touch()
        assert gm.scan_pngs(tmp_path) == ["image.png"]

    def test_recursive_results_are_sorted_forward_slash_paths_relative_to_root(self, tmp_path):
        for relative in ("zebra.png", "apple.png", "animals/cat.png", "a/b/img.png"):
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        assert gm.scan_pngs(tmp_path) == ["a/b/img.png", "animals/cat.png", "apple.png", "zebra.png"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_both_dirs(base: Path, filenames: list[str] | None = None) -> tuple[Path, Path]:
    """Create <base>/images/pre_shine/ and <base>/images/post_shine/.

    If *filenames* is given, creates those PNG stubs in both directories.
    Returns (pre_dir, post_dir).
    """
    pre  = base / "images" / "pre_shine"
    post = base / "images" / "post_shine"
    pre.mkdir(parents=True)
    post.mkdir(parents=True)
    for name in (filenames or []):
        (pre  / name).touch()
        (post / name).touch()
    return pre, post


@pytest.fixture
def run_manifest(task_env: Path):
    """Build both image dirs (+ optional shared dir), write the config, run main(), return the parsed manifest."""
    def _run(filenames: list[str] | None = None, shared_filenames: list[str] | None = None, **config) -> dict:
        _make_both_dirs(task_env, ["img.png"] if filenames is None else filenames)
        config["main_root"] = str(task_env / "images")
        if shared_filenames is not None:
            shared = task_env / "shared"
            shared.mkdir()
            for name in shared_filenames:
                (shared / name).touch()
            config["stimuli_path"] = str(shared)
        make_config(task_env, **config)
        gm.main()
        return json.loads((task_env / "stimuli_manifest.json").read_text())
    return _run


# ---------------------------------------------------------------------------
# main — config / file-system errors
# ---------------------------------------------------------------------------

class TestMainErrors:
    def test_missing_config_raises_file_not_found(self, task_env):
        with pytest.raises(FileNotFoundError, match="task_config.json"):
            gm.main()

    def test_empty_main_root_raises_value_error(self, task_env):
        make_config(task_env)  # main_root="" by default
        with pytest.raises(ValueError, match="stimuli_path"):
            gm.main()

    def test_nonexistent_pre_dir_raises(self, task_env):
        make_config(task_env, main_root=str(task_env / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="pre_shine"):
            gm.main()

    def test_nonexistent_post_dir_raises(self, task_env):
        # pre_shine exists and has images; post_shine is absent.
        pre = task_env / "images" / "pre_shine"
        pre.mkdir(parents=True)
        (pre / "img.png").touch()
        make_config(task_env, main_root=str(task_env / "images"))
        with pytest.raises(FileNotFoundError, match="post_shine"):
            gm.main()

    def test_empty_pre_dir_raises(self, task_env):
        # Both dirs exist but pre_shine has no PNGs.
        pre  = task_env / "images" / "pre_shine"
        post = task_env / "images" / "post_shine"
        pre.mkdir(parents=True)
        post.mkdir(parents=True)
        make_config(task_env, main_root=str(task_env / "images"))
        with pytest.raises(RuntimeError, match="empty"):
            gm.main()


# ---------------------------------------------------------------------------
# main — manifest content
# ---------------------------------------------------------------------------

class TestMainManifestContent:
    def test_all_three_sets_written_when_paths_valid(self, run_manifest):
        manifest = run_manifest(shared_filenames=["img.png"])
        assert set(manifest.keys()) == {"images", "practice_images", "catch_images"}
        assert manifest["practice_images"] == manifest["catch_images"]  # same shared dir

    def test_main_images_only_when_no_shared_path_and_no_shine_variant_key(self, run_manifest):
        manifest = run_manifest(["cat.png", "dog.png"])
        assert manifest["images"] == ["cat.png", "dog.png"]
        assert "shine_variant" not in manifest
        assert "practice_images" not in manifest
        assert "catch_images" not in manifest


# ---------------------------------------------------------------------------
# main — post_shine verification
# ---------------------------------------------------------------------------

class TestPostShineVerification:
    def test_matching_dirs_produce_no_differ_warning(self, run_manifest, capsys):
        run_manifest(["cat.png", "dog.png"])
        out = capsys.readouterr().out
        assert "differ" not in out
        assert "verified OK" in out

    def test_diverging_dirs_produce_warning(self, task_env, capsys):
        pre, post = _make_both_dirs(task_env, ["shared.png"])
        (pre  / "only_in_pre.png").touch()
        (post / "only_in_post.png").touch()
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "differ" in out


# ---------------------------------------------------------------------------
# main — validation warnings
# ---------------------------------------------------------------------------

class TestMainValidation:
    @pytest.mark.parametrize("set_name, n_shared, images_per_trial, expect_warning", [
        ("practice_images", 1, 8, True),    # 1 image below the threshold of 8
        ("catch_images", 1, 20, True),      # 1 image below the threshold of 20
        ("practice_images", 8, 8, False),
        ("catch_images", 20, 20, False),
    ])
    def test_warns_only_when_shared_set_is_below_threshold(
        self, run_manifest, capsys, set_name, n_shared, images_per_trial, expect_warning,
    ):
        run_manifest(shared_filenames=[f"img_{i}.png" for i in range(n_shared)], images_per_trial=images_per_trial)
        out = capsys.readouterr().out
        assert (f"'{set_name}' has" in out) == expect_warning
        if expect_warning:
            assert "WARNING" in out
