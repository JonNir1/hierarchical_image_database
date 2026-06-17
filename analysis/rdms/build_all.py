"""
CLI to build any or all reference RDMs.

Usage (from repo root, with .venv active):

    # Build everything
    python -m analysis.rdms.build_all

    # Build a specific subset
    python -m analysis.rdms.build_all --only sens_pre sem_km

    # Build everything except the slow ones
    python -m analysis.rdms.build_all --skip sem_wn clip_pre clip_post

Available targets:
    sens_pre   -- D_sens_pre.npy   (pixel Euclidean, pre-SHINE)
    sens_post  -- D_sens_post.npy  (pixel Euclidean, post-SHINE)
    sem_km     -- D_sem_km.npy     (Kiani-Mur tree-edge distance)
    sem_wn     -- D_sem_wn.npy     (WordNet shortest-path via manifest synsets)
    clip_pre   -- D_clip_pre.npy   (CLIP ViT-B/32 cosine, pre-SHINE)
    clip_post  -- D_clip_post.npy  (CLIP ViT-B/32 cosine, post-SHINE)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback

# name -> (module, function, positional_args)
_BUILDERS: dict[str, tuple[str, str, list]] = {
    "sens_pre":  ("analysis.rdms.sensory",      "build_sensory_rdm",  ["pre_shine"]),
    "sens_post": ("analysis.rdms.sensory",      "build_sensory_rdm",  ["post_shine"]),
    "sem_km":    ("analysis.rdms.semantic_km",  "build_km_rdm",       []),
    "sem_wn":    ("analysis.rdms.semantic_wn_dir", "build_wn_rdm",     []),
    "clip_pre":  ("analysis.rdms.clip",         "build_clip_rdm",     ["pre_shine"]),
    "clip_post": ("analysis.rdms.clip",         "build_clip_rdm",     ["post_shine"]),
}

_ALL_NAMES = list(_BUILDERS.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reference RDMs for the hierarchical image dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="Build only these RDMs (space-separated).",
    )
    group.add_argument(
        "--skip", nargs="+", metavar="NAME",
        help="Build everything except these RDMs.",
    )
    args = parser.parse_args()

    # Validate names
    unknown = set(args.only or args.skip or []) - set(_ALL_NAMES)
    if unknown:
        parser.error(f"Unknown RDM name(s): {sorted(unknown)}. Choose from: {_ALL_NAMES}")

    targets = _ALL_NAMES
    if args.only:
        targets = args.only
    elif args.skip:
        targets = [t for t in targets if t not in args.skip]

    print(f"Targets: {targets}\n")
    failures: list[str] = []
    for name in targets:
        module_name, func_name, func_args = _BUILDERS[name]
        print(f"{'=' * 50}")
        print(f"Building: {name}")
        print(f"{'=' * 50}")
        try:
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            func(*func_args)
            print(f"  OK: {name}\n")
        except Exception as exc:  # noqa: BLE001 — intentional: CLI must survive any builder error
            print(f"  FAILED: {name}: {exc}")
            traceback.print_exc()
            failures.append(name)
            print()

    print(f"{'=' * 50}")
    if failures:
        print(f"FAILED ({len(failures)}/{len(targets)}): {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"All {len(targets)} builder(s) completed successfully.")


if __name__ == "__main__":
    main()
