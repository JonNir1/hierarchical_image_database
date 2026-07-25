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

    def test_finds_png_files_in_root(self, tmp_path):
        (tmp_path / "cat.png").touch()
        (tmp_path / "dog.png").touch()
        assert gm.scan_pngs(tmp_path) == ["cat.png", "dog.png"]

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

    def test_recursive_into_subdirectories(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.png").touch()
        (sub / "nested.png").touch()
        result = gm.scan_pngs(tmp_path)
        assert "root.png" in result
        assert "subdir/nested.png" in result

    def test_paths_use_forward_slashes(self, tmp_path):
        sub = tmp_path / "animals"
        sub.mkdir()
        (sub / "cat.png").touch()
        result = gm.scan_pngs(tmp_path)
        assert result == ["animals/cat.png"]
        assert "\\" not in result[0]

    def test_results_are_sorted_alphabetically(self, tmp_path):
        for name in ("zebra.png", "apple.png", "mango.png"):
            (tmp_path / name).touch()
        assert gm.scan_pngs(tmp_path) == ["apple.png", "mango.png", "zebra.png"]

    def test_paths_relative_to_directory_root(self, tmp_path):
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "img.png").touch()
        result = gm.scan_pngs(tmp_path)
        assert result == ["a/b/img.png"]


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
    def test_all_three_sets_written_when_paths_valid(self, task_env):
        _make_both_dirs(task_env, ["img.png"])
        shared = task_env / "shared"
        shared.mkdir()
        (shared / "img.png").touch()
        make_config(
            task_env,
            main_root=str(task_env / "images"),
            stimuli_path=str(shared),
        )
        gm.main()
        manifest = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert set(manifest.keys()) == {"images", "practice_images", "catch_images"}
        assert manifest["practice_images"] == manifest["catch_images"]  # same shared dir

    def test_no_shine_variant_key_in_manifest(self, task_env):
        _make_both_dirs(task_env, ["img.png"])
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        manifest = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert "shine_variant" not in manifest

    def test_image_filenames_stored_in_manifest(self, task_env):
        _make_both_dirs(task_env, ["cat.png", "dog.png"])
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        manifest = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert manifest["images"] == ["cat.png", "dog.png"]

    def test_skipped_set_omitted_from_manifest(self, task_env):
        _make_both_dirs(task_env, ["img.png"])
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        manifest = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert "practice_images" not in manifest
        assert "catch_images" not in manifest

    def test_manifest_written_to_correct_file(self, task_env):
        _make_both_dirs(task_env, ["img.png"])
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        assert (task_env / "stimuli_manifest.json").exists()

    def test_manifest_is_valid_json_object(self, task_env):
        _make_both_dirs(task_env, ["img.png"])
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        data = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# main — post_shine verification
# ---------------------------------------------------------------------------

class TestPostShineVerification:
    def test_matching_dirs_produce_no_differ_warning(self, task_env, capsys):
        _make_both_dirs(task_env, ["cat.png", "dog.png"])
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
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
    def test_warns_when_too_few_practice_images(self, task_env, capsys):
        _make_both_dirs(task_env, ["img.png"])
        shared_dir = task_env / "shared"
        shared_dir.mkdir()
        (shared_dir / "a.png").touch()  # 1 image, practice threshold is 8
        make_config(task_env,
                    main_root=str(task_env / "images"),
                    stimuli_path=str(shared_dir),
                    images_per_trial=8)
        gm.main()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "practice_images" in out

    def test_warns_when_too_few_catch_images(self, task_env, capsys):
        _make_both_dirs(task_env, ["img.png"])
        shared_dir = task_env / "shared"
        shared_dir.mkdir()
        (shared_dir / "a.png").touch()  # 1 image, catch threshold is 20
        make_config(task_env,
                    main_root=str(task_env / "images"),
                    stimuli_path=str(shared_dir),
                    images_per_trial=20)
        gm.main()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "catch_images" in out

    def test_no_practice_warning_when_count_meets_threshold(self, task_env, capsys):
        _make_both_dirs(task_env, ["img.png"])
        shared_dir = task_env / "shared"
        shared_dir.mkdir()
        for i in range(8):
            (shared_dir / f"img_{i}.png").touch()
        make_config(task_env,
                    main_root=str(task_env / "images"),
                    stimuli_path=str(shared_dir),
                    images_per_trial=8)
        gm.main()
        out = capsys.readouterr().out
        assert "practice_images' has" not in out  # no threshold warning

    def test_no_catch_warning_when_count_meets_threshold(self, task_env, capsys):
        _make_both_dirs(task_env, ["img.png"])
        shared_dir = task_env / "shared"
        shared_dir.mkdir()
        for i in range(20):
            (shared_dir / f"img_{i}.png").touch()
        make_config(task_env,
                    main_root=str(task_env / "images"),
                    stimuli_path=str(shared_dir),
                    images_per_trial=20)
        gm.main()
        out = capsys.readouterr().out
        assert "catch_images' has" not in out
