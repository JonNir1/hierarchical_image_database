import json
import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "task_config.json"
MANIFEST_PATH = TASK_DIR / "stimuli_manifest.json"

# Maps (config section, config key) → manifest key
SETS = [
    ("stimuli_paths", "main",     "images"),
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
    """Read config.json, scan each stimulus directory, and write stimuli_manifest.json.

    Produces a JSON file with three keys (``images``, ``practice_images``,
    ``catch_images``), each containing a sorted list of POSIX-relative paths for
    the corresponding stimulus set.  Missing or empty directories emit warnings
    but do not abort — the manifest is still written with whatever sets were
    found, so partial runs remain usable during development.

    Validates that each set contains at least as many images as the matching
    threshold in config.json and warns if not.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"task_config.json not found at {CONFIG_PATH}")

    with CONFIG_PATH.open() as fh:
        config = json.load(fh)

    stimuli_paths = config.get("stimuli_paths", {})
    design        = config.get("design", {})
    catch_trials  = config.get("catch_trials", {})

    manifest: dict[str, list[str]] = {}

    for section_key, path_key, manifest_key in SETS:
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
    if "images" in manifest and len(manifest["images"]) < 1:
        print("WARNING: 'images' set has fewer than 1 image.")

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
