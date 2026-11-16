"""Tests for analysis.pre_vs_post_shine.ground_truth: overlap matching and the tracked matches file."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.build_rdms.common import load_manifest
from analysis.pre_vs_post_shine import ground_truth as gt


def _blob(rng: np.random.Generator) -> np.ndarray:
    img = np.full((175, 175, 3), 128, dtype=np.uint8)
    y, x = rng.integers(20, 100, size=2)
    img[y:y + 50, x:x + 50] = rng.integers(0, 255, size=3)
    return img


def test_matching_is_one_to_one_leaves_empty_categories_unmatched_and_the_tracked_file_is_valid(monkeypatch):
    rng = np.random.default_rng(1)
    images = [_blob(rng) for _ in range(4)]
    stimuli = gt.Stimuli(images + [_blob(rng)], ["natural inanimate"] * 4 + ["human face"])
    manifest = pd.DataFrame({"category": ["NO"] * 5})
    ours = {row: images[3 - row] if row < 4 else _blob(rng) for row in range(5)}  # reversed order plus a decoy
    monkeypatch.setattr(gt, "image_paths", lambda variant: list(range(5)))
    monkeypatch.setattr(gt, "load_our_image", lambda path: ours[path])

    result = gt.match_by_phash(stimuli, manifest, manifest.index).set_index("ext_index")

    assert result.loc[:3, "manifest_row"].tolist() == [3, 2, 1, 0] and (result.loc[:3, "hamming"] == 0).all()
    assert np.isnan(result.loc[4, "manifest_row"])  # no human-face candidate

    overlap = gt.load_overlap()  # the reviewed file: 42 matches, every path in the manifest, manifest order
    assert len(overlap) == 42 and overlap["manifest_row"].is_monotonic_increasing
    assert set(overlap["curated_path"]) <= {p.replace("\\", "/") for p in load_manifest()["curated_path"]}
