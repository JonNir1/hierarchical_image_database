"""Builders and patch helpers shared by the s0_build_rdms test modules.

Imported as `from _rdm_helpers import ...` (pytest puts this directory on sys.path).
Fixtures stay in the test modules; only plain functions live here.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import analysis.s0_build_rdms.common as common

_BACKSLASH = chr(92)


def patch_common_results(monkeypatch: pytest.MonkeyPatch, results_dir, n: int) -> None:
    """Redirect common.RESULTS_DIR and shrink the expected N so tests skip the real 725-image set."""
    monkeypatch.setattr(common, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(common, "_EXPECTED_LEN", n * (n - 1) // 2)
    monkeypatch.setattr(common, "_EXPECTED_N", n)


def synthetic_manifest(paths: list[str]) -> pd.DataFrame:
    """Manifest frame with the columns the RDM builders read, for Windows-style curated paths."""
    return pd.DataFrame({
        "curated_path": paths,
        "curated_filename": [p.split(_BACKSLASH)[-1] for p in paths],
        "category": ["X"] * len(paths),
        "wn_synset_name": ["dog.n.01"] * len(paths),
        "source_dataset": ["test"] * len(paths),
        "source_filename_or_url": [""] * len(paths),
        "manual_match_validation": ["V"] * len(paths),
    })


def pairwise_json(pairs: dict[tuple[str, str], float], variant_dir: str = "pre_shine") -> str:
    """Serialise {(img_a, img_b): distance} the way the SpAM task stores a trial's pairwise_distances."""
    items = [
        {"src1": f"./images/{variant_dir}/{a}", "src2": f"./images/{variant_dir}/{b}", "distance": d}
        for (a, b), d in pairs.items()
    ]
    return json.dumps(items)
