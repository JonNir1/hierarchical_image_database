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
        with patch.object(gm, "TASK_DIR", tmp_path):
            assert gm.resolve_path(str(target)) == target

    def test_relative_path_resolved_against_task_dir(self, tmp_path):
        with patch.object(gm, "TASK_DIR", tmp_path):
            result = gm.resolve_path("stimuli")
        assert result == (tmp_path / "stimuli").resolve()

    def test_relative_path_with_subdirs(self, tmp_path):
        with patch.object(gm, "TASK_DIR", tmp_path):
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
# main — config / file-system errors
# ---------------------------------------------------------------------------

class TestMainErrors:
    def test_missing_config_raises_file_not_found(self, task_env):
        with pytest.raises(FileNotFoundError, match="task_config.json"):
            gm.main()

    def test_empty_main_root_raises_value_error(self, task_env):
        make_config(task_env)  # main_root="" by default
        with pytest.raises(ValueError, match="main_root"):
            gm.main()

    def test_invalid_shine_variant_raises_value_error(self, task_env):
        make_config(task_env, main_root="images", shine_variant="bad")
        with pytest.raises(ValueError, match="shine_variant"):
            gm.main()

    def test_nonexistent_main_dir_raises(self, task_env):
        make_config(task_env, main_root=str(task_env / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="Main stimulus directory"):
            gm.main()

    def test_empty_main_dir_raises(self, task_env):
        # Create the variant subdir but leave it empty.
        (task_env / "images" / "pre_shine").mkdir(parents=True)
        make_config(task_env, main_root=str(task_env / "images"))
        with pytest.raises(RuntimeError, match="empty"):
            gm.main()


# ---------------------------------------------------------------------------
# main — manifest content
# ---------------------------------------------------------------------------

def _make_main_dir(base: Path, variant: str = "pre") -> Path:
    """Create <base>/images/<variant>_shine/ and return the resolved variant dir."""
    d = base / "images" / f"{variant}_shine"
    d.mkdir(parents=True)
    return d


class TestMainManifestContent:
    def test_all_three_sets_written_when_paths_valid(self, task_env):
        main_dir = _make_main_dir(task_env)
        (main_dir / "img.png").touch()
        for name in ("practice", "catch"):
            d = task_env / name
            d.mkdir()
            (d / "img.png").touch()
        make_config(
            task_env,
            main_root=str(task_env / "images"),
            stimuli_practice_path=str(task_env / "practice"),
            stimuli_catch_path=str(task_env / "catch"),
        )
        gm.main()
        manifest = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert set(manifest.keys()) == {
            "shine_variant", "images", "practice_images", "catch_images"
        }

    def test_image_filenames_stored_in_manifest(self, task_env):
        main_dir = _make_main_dir(task_env)
        (main_dir / "cat.png").touch()
        (main_dir / "dog.png").touch()
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        manifest = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert manifest["images"] == ["cat.png", "dog.png"]

    def test_shine_variant_written_to_manifest(self, task_env):
        main_dir = _make_main_dir(task_env, variant="post")
        (main_dir / "img.png").touch()
        make_config(task_env, main_root=str(task_env / "images"),
                    shine_variant="post")
        gm.main()
        manifest = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert manifest["shine_variant"] == "post"

    def test_skipped_set_omitted_from_manifest(self, task_env):
        main_dir = _make_main_dir(task_env)
        (main_dir / "img.png").touch()
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        manifest = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert "practice_images" not in manifest
        assert "catch_images" not in manifest

    def test_manifest_written_to_correct_file(self, task_env):
        main_dir = _make_main_dir(task_env)
        (main_dir / "img.png").touch()
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        assert (task_env / "stimuli_manifest.json").exists()

    def test_manifest_is_valid_json_object(self, task_env):
        main_dir = _make_main_dir(task_env)
        (main_dir / "img.png").touch()
        make_config(task_env, main_root=str(task_env / "images"))
        gm.main()
        data = json.loads((task_env / "stimuli_manifest.json").read_text())
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# main — validation warnings
# ---------------------------------------------------------------------------

class TestMainValidation:
    def test_warns_when_too_few_practice_images(self, task_env, capsys):
        main_dir = _make_main_dir(task_env)
        (main_dir / "img.png").touch()
        practice_dir = task_env / "practice"
        practice_dir.mkdir()
        (practice_dir / "a.png").touch()  # 1 image, threshold is 8
        make_config(task_env,
                    main_root=str(task_env / "images"),
                    stimuli_practice_path=str(practice_dir),
                    practice_images_per_trial=8)
        gm.main()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "practice_images" in out

    def test_warns_when_too_few_catch_images(self, task_env, capsys):
        main_dir = _make_main_dir(task_env)
        (main_dir / "img.png").touch()
        catch_dir = task_env / "catch"
        catch_dir.mkdir()
        (catch_dir / "a.png").touch()  # 1 image, threshold is 20
        make_config(task_env,
                    main_root=str(task_env / "images"),
                    stimuli_catch_path=str(catch_dir),
                    images_per_trial=20)
        gm.main()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "catch_images" in out

    def test_no_practice_warning_when_count_meets_threshold(self, task_env, capsys):
        main_dir = _make_main_dir(task_env)
        (main_dir / "img.png").touch()
        practice_dir = task_env / "practice"
        practice_dir.mkdir()
        for i in range(8):
            (practice_dir / f"img_{i}.png").touch()
        make_config(task_env,
                    main_root=str(task_env / "images"),
                    stimuli_practice_path=str(practice_dir),
                    practice_images_per_trial=8)
        gm.main()
        out = capsys.readouterr().out
        assert "practice_images' has" not in out  # no threshold warning

    def test_no_catch_warning_when_count_meets_threshold(self, task_env, capsys):
        main_dir = _make_main_dir(task_env)
        (main_dir / "img.png").touch()
        catch_dir = task_env / "catch"
        catch_dir.mkdir()
        for i in range(20):
            (catch_dir / f"img_{i}.png").touch()
        make_config(task_env,
                    main_root=str(task_env / "images"),
                    stimuli_catch_path=str(catch_dir),
                    images_per_trial=20)
        gm.main()
        out = capsys.readouterr().out
        assert "catch_images' has" not in out
