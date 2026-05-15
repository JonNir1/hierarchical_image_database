import json
import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "task_config.json"
MANIFEST_PATH = TASK_DIR / "stimuli_manifest.json"

VALID_SHINE_VARIANTS = ("pre", "post")

# Practice and catch sets are variant-agnostic and resolved directly from config.
# The main set is resolved dynamically from (stimuli_paths.main_root, shine.shine_variant)
# — see `resolve_main_dir`.
SECONDARY_SETS = [
    ("stimuli_paths", "practice", "practice_images"),
    ("stimuli_paths", "catch",    "catch_images"),
]


def resolve_path(raw: str) -> Path:
    """Resolve a config path string to an absolute Path.

    Absolute paths are returned unchanged. Relative paths are resolved
    relative to the SpAM_Task directory (where this script lives), not the
    current working directory, so the script can be run from anywhere.
    """
    p = Path(raw)
    return p if p.is_absolute() else (TASK_DIR / p).resolve()


def resolve_main_dir(config: dict) -> tuple[Path, str]:
    """Resolve the main-stimulus directory by combining `stimuli_paths.main_root`
    and `shine.shine_variant` into `<main_root>/<variant>_shine/`.

    Returns (resolved_directory, variant_string). Raises ValueError on invalid
    or missing config values.
    """
    main_root_raw = config.get("stimuli_paths", {}).get("main_root", "")
    if not main_root_raw:
        raise ValueError("'stimuli_paths.main_root' is missing or empty in task_config.json")

    variant = config.get("shine", {}).get("shine_variant", "")
    if variant not in VALID_SHINE_VARIANTS:
        raise ValueError(
            f"'shine.shine_variant' must be one of {VALID_SHINE_VARIANTS}; "
            f"got {variant!r}"
        )

    main_dir = resolve_path(main_root_raw) / f"{variant}_shine"
    return main_dir, variant


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

    Produces a JSON file with the keys ``shine_variant``, ``images``,
    ``practice_images``, and ``catch_images``. The active SHINE variant is
    recorded in the manifest so task.js and downstream consumers can sanity-check
    it.

    The main set is fatal: if the variant subdirectory is missing or empty,
    the script aborts. Practice and catch directories emit warnings on
    missing/empty content but do not abort, so partial runs remain usable during
    development.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"task_config.json not found at {CONFIG_PATH}")

    with CONFIG_PATH.open() as fh:
        config = json.load(fh)

    design       = config.get("design", {})
    catch_trials = config.get("catch_trials", {})

    main_dir, variant = resolve_main_dir(config)

    manifest: dict[str, object] = {"shine_variant": variant}

    # Main set — fatal on missing/empty.
    if not main_dir.exists():
        raise FileNotFoundError(
            f"Main stimulus directory does not exist: {main_dir}\n"
            f"Populate {variant}_shine/ under stimuli_paths.main_root and retry."
        )
    main_images = scan_pngs(main_dir)
    if not main_images:
        raise RuntimeError(
            f"Main stimulus directory is empty (no .png files): {main_dir}"
        )
    manifest["images"] = main_images
    print(f"images ({variant}-SHINE): {len(main_images)} images in {main_dir}")

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
