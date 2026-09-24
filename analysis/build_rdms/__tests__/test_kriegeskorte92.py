"""Tests for analysis.build_rdms.kriegeskorte92: hashing and one-to-one matching on synthetic images."""
from __future__ import annotations

import numpy as np
import pandas as pd

import analysis.build_rdms.kriegeskorte92 as k92


def _blob(rng: np.random.Generator) -> np.ndarray:
    img = np.full((175, 175, 3), 128, dtype=np.uint8)
    y, x = rng.integers(20, 100, size=2)
    img[y:y + 50, x:x + 50] = rng.integers(0, 255, size=3)
    return img


def test_average_hash_is_identical_for_identical_images_and_differs_otherwise():
    rng = np.random.default_rng(0)
    a, b = _blob(rng), _blob(rng)
    assert (k92.average_hash(a) == k92.average_hash(a.copy())).all()
    assert (k92.average_hash(a) != k92.average_hash(b)).any()


def test_match_by_phash_assigns_true_pairs_one_to_one(monkeypatch):
    rng = np.random.default_rng(1)
    images = [_blob(rng) for _ in range(4)]
    stimuli = k92.Stimuli(images, ["natural inanimate"] * 4, ["NO"] * 4)
    manifest = pd.DataFrame({"category": ["NO"] * 5})
    ours = {row: images[3 - row] if row < 4 else _blob(rng) for row in range(5)}
    monkeypatch.setattr(k92, "image_paths", lambda variant: list(range(5)))
    monkeypatch.setattr(k92, "load_our_image", lambda path: ours[path])

    result = k92.match_by_phash(stimuli, manifest, manifest.index)

    assert result["manifest_row"].nunique() == 4
    assert result.set_index("ext_index")["manifest_row"].to_dict() == {0: 3, 1: 2, 2: 1, 3: 0}
    assert (result["hamming"] == 0).all()


def test_match_by_phash_leaves_categories_without_candidates_unmatched(monkeypatch):
    rng = np.random.default_rng(2)
    stimuli = k92.Stimuli([_blob(rng)], ["human face"], ["HF"])
    manifest = pd.DataFrame({"category": ["NO"]})
    monkeypatch.setattr(k92, "image_paths", lambda variant: [0])
    monkeypatch.setattr(k92, "load_our_image", lambda path: _blob(rng))
    result = k92.match_by_phash(stimuli, manifest, manifest.index)
    assert result["manifest_row"].isna().all()
