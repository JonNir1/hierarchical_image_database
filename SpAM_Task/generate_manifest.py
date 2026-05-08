import argparse
import json
import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "config.json"
MANIFEST_PATH = TASK_DIR / "stimuli_manifest.json"

SETS = [
    ("stimuli_path",          "images"),
    ("stimuli_practice_path", "practice_images"),
    ("stimuli_catch_path",    "catch_images"),
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


def parse_args() -> argparse.Namespace:
    """Parse optional CLI path overrides.

    CLI arguments let you point generate_manifest.py at local filesystem paths
    that differ from the URL prefixes stored in config.json (e.g. when stimuli
    live outside the served directory during local development).  config.json is
    never modified — the overrides only affect which directories are scanned.

    Example — main stimuli stored outside SpAM_Task/:
        python generate_manifest.py --stimuli "C:/Users/me/Desktop/MyImages"

    When an argument is omitted the corresponding config.json value is used
    unchanged, so existing usage (no arguments) is fully backward-compatible.
    """
    parser = argparse.ArgumentParser(
        description="Scan stimulus directories and write stimuli_manifest.json.",
    )
    parser.add_argument(
        "--stimuli", metavar="PATH", default=None,
        help="Override stimuli_path from config.json with a local filesystem path",
    )
    parser.add_argument(
        "--practice", metavar="PATH", default=None,
        help="Override stimuli_practice_path from config.json",
    )
    parser.add_argument(
        "--catch", metavar="PATH", default=None,
        help="Override stimuli_catch_path from config.json",
    )
    return parser.parse_args()


def main() -> None:
    """Read config.json, scan each stimulus directory, and write stimuli_manifest.json.

    Produces a JSON file with three keys (``images``, ``practice_images``,
    ``catch_images``), each containing a sorted list of POSIX-relative paths for
    the corresponding stimulus set.  Missing or empty directories emit warnings
    but do not abort — the manifest is still written with whatever sets were
    found, so partial runs remain usable during development.

    Validates that each set contains at least as many images as the matching
    ``*_per_trial`` threshold in config.json and warns if not.
    """
    args = parse_args()

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")

    with CONFIG_PATH.open() as fh:
        config = json.load(fh)

    # Apply CLI overrides — only affects directory scanning, not config.json.
    cli_overrides = {
        "stimuli_path":          args.stimuli,
        "stimuli_practice_path": args.practice,
        "stimuli_catch_path":    args.catch,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            config[key] = val

    manifest: dict[str, list[str]] = {}

    for config_key, manifest_key in SETS:
        raw = config.get(config_key, "")
        if not raw:
            print(f"WARNING: '{config_key}' is missing or empty — skipping {manifest_key}.")
            continue

        directory = resolve_path(raw)
        if not directory.exists():
            print(f"WARNING: '{config_key}' path does not exist ({directory}) — skipping {manifest_key}.")
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
        threshold = config.get("practice_images_per_trial", 0)
        n = len(manifest["practice_images"])
        if n < threshold:
            print(
                f"WARNING: 'practice_images' has {n} image(s) but config requires "
                f"at least {threshold} (practice_images_per_trial)."
            )

    if "catch_images" in manifest:
        threshold = config.get("catch_images_per_trial", 0)
        n = len(manifest["catch_images"])
        if n < threshold:
            print(
                f"WARNING: 'catch_images' has {n} emoji image(s) but config requires "
                f"at least {threshold} (catch_images_per_trial)."
            )

    with MANIFEST_PATH.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
