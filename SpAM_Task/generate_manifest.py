import json
import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
CONFIG_PATH = TASK_DIR / "task_config.json"
MANIFEST_PATH = TASK_DIR / "stimuli_manifest.json"

# Practice and catch sets are variant-agnostic and resolved directly from config.
# Both are scanned from catch_trials.stimuli_path -- practice has no separate path,
# it reuses the catch directory. The main set is resolved from design.stimuli_path;
# both pre_shine/ and post_shine/ subdirectories are scanned and verified to be identical.
SECONDARY_SETS = [
    ("catch_trials", "stimuli_path", "practice_images"),
    ("catch_trials", "stimuli_path", "catch_images"),
]


def resolve_path(raw: str) -> Path:
    """Resolve a config path string to an absolute Path.

    Absolute paths are returned unchanged. Relative paths are resolved
    relative to the project root (parent of SpAM_Task/), matching how the
    browser resolves the same strings relative to the page at <root>/index.html.
    This way the same path string in task_config.json works for both Python
    (manifest generation) and the browser (image fetching).
    """
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def resolve_main_root(config: dict) -> Path:
    """Resolve the main-stimulus root directory from experimental_trials.stimuli_path.

    Returns the resolved Path. Raises ValueError if the key is missing or empty.
    """
    main_root_raw = config.get("experimental_trials", {}).get("stimuli_path", "")
    if not main_root_raw:
        raise ValueError("'experimental_trials.stimuli_path' is missing or empty in task_config.json")
    return resolve_path(main_root_raw)


def scan_pngs(directory: Path) -> list[str]:
    """Recursively find all .png files under *directory*.

    Returns a sorted list of POSIX-style paths relative to *directory*
    (e.g. ``"subdir/image.png"``). Sorting ensures a deterministic manifest
    regardless of filesystem traversal order.
    """
    results = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".png"):
                rel = Path(root, f).relative_to(directory)
                results.append(rel.as_posix())
    return sorted(results)


def main() -> None:
    """Read task_config.json, scan each stimulus directory, and write stimuli_manifest.json.

    Produces a JSON file with the keys ``images``, ``practice_images``, and
    ``catch_images``. ``images`` contains filenames relative to the variant root
    (identical for pre_shine/ and post_shine/ since the two directories mirror each
    other). Both variant directories are scanned: pre_shine/ provides the canonical
    file list (fatal if missing/empty), and post_shine/ is verified to contain the
    same filenames (fatal if the directory is missing; warning if the file sets diverge).

    Practice and catch directories emit warnings on missing/empty content but do not
    abort, so partial runs remain usable during development.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"task_config.json not found at {CONFIG_PATH}")

    with CONFIG_PATH.open() as fh:
        config = json.load(fh)

    design       = config.get("design", {})
    catch_trials = config.get("catch_trials", {})

    main_root = resolve_main_root(config)
    pre_dir   = main_root / "pre_shine"
    post_dir  = main_root / "post_shine"

    manifest: dict[str, object] = {}

    # pre_shine/ — canonical image list, fatal on missing/empty.
    if not pre_dir.exists():
        raise FileNotFoundError(
            f"Main stimulus directory does not exist: {pre_dir}\n"
            f"Populate pre_shine/ under design.stimuli_path and retry."
        )
    pre_images = scan_pngs(pre_dir)
    if not pre_images:
        raise RuntimeError(
            f"Main stimulus directory is empty (no .png files): {pre_dir}"
        )
    manifest["images"] = pre_images
    print(f"images: {len(pre_images)} images in {pre_dir}")

    # post_shine/ — must exist and match pre_shine/ file list.
    if not post_dir.exists():
        raise FileNotFoundError(
            f"post_shine directory does not exist: {post_dir}\n"
            f"Populate post_shine/ under design.stimuli_path and retry."
        )
    post_images = scan_pngs(post_dir)
    if set(pre_images) != set(post_images):
        print(
            f"WARNING: pre_shine and post_shine file sets differ.\n"
            f"  pre only:  {sorted(set(pre_images) - set(post_images))[:5]}\n"
            f"  post only: {sorted(set(post_images) - set(pre_images))[:5]}"
        )
    else:
        print(f"  post_shine: verified OK ({len(post_images)} images in {post_dir})")

    # Secondary sets — warn on missing/empty.
    for section_key, path_key, manifest_key in SECONDARY_SETS:
        raw = config.get(section_key, {}).get(path_key, "")
        config_ref = f"{section_key}.{path_key}"
        if not raw:
            print(f"WARNING: '{config_ref}' is missing or empty — skipping {manifest_key}.")
            continue

        directory = resolve_path(raw)
        if not directory.exists():
            print(f"WARNING: '{config_ref}' path does not exist ({directory}) — skipping {manifest_key}.")
            continue

        images = scan_pngs(directory)
        manifest[manifest_key] = images
        n = len(images)
        print(f"{manifest_key}: {n} image{'s' if n != 1 else ''} in {directory}")
        if n == 0:
            print(f"WARNING: No .png images found in {directory}.")

    # Validation against config thresholds
    if "practice_images" in manifest:
        threshold = design.get("practice_images_per_trial", 0)
        n = len(manifest["practice_images"])
        if n < threshold:
            print(
                f"WARNING: 'practice_images' has {n} image(s) but config requires "
                f"at least {threshold} (design.practice_images_per_trial)."
            )

    if "catch_images" in manifest:
        threshold = catch_trials.get("images_per_trial", 0)
        n = len(manifest["catch_images"])
        if n < threshold:
            print(
                f"WARNING: 'catch_images' has {n} image(s) but config requires "
                f"at least {threshold} (catch_trials.images_per_trial)."
            )

    with MANIFEST_PATH.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
