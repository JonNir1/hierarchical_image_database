import json
import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "config.json"
MANIFEST_PATH = TASK_DIR / "stimuli_manifest.json"

SETS = [
    ("stimuli_practice_path", "practice_images"),
]


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (TASK_DIR / p).resolve()


def scan_pngs(directory: Path) -> list[str]:
    results = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".png"):
                rel = Path(root, f).relative_to(directory)
                results.append(rel.as_posix())
    return sorted(results)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")

    with CONFIG_PATH.open() as fh:
        config = json.load(fh)

    manifest: dict[str, list[str]] = {}

    # Main image pool: split into catch_images (first N) and images (remainder).
    # The catch pool is fixed across participants; the main pool is randomised
    # per-participant by buildTrialLists. Keeping them disjoint at manifest level
    # guarantees catch images never appear in a participant's main trials.
    raw_stimuli = config.get("stimuli_path", "")
    if raw_stimuli:
        main_dir    = resolve_path(raw_stimuli)
        catch_n     = config.get("catch_pool_size", 20)
        if main_dir.exists():
            all_images          = scan_pngs(main_dir)
            manifest["catch_images"] = all_images[:catch_n]
            manifest["images"]       = all_images[catch_n:]
            print(f"images: {len(manifest['images'])} (catch pool: {catch_n}) in {main_dir}")
            if len(all_images) <= catch_n:
                print(f"WARNING: catch_pool_size ({catch_n}) >= total images ({len(all_images)}). "
                      "Main image pool will be empty.")
        else:
            print(f"WARNING: 'stimuli_path' does not exist ({main_dir}) — skipping images.")
    else:
        print("WARNING: 'stimuli_path' is empty — skipping images.")

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
                f"WARNING: 'catch_images' has {n} image(s) but config requires "
                f"at least {threshold} (catch_images_per_trial)."
            )

    with MANIFEST_PATH.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
