"""Tests for analysis.rdms.common, analysis.rdms.semantic_km (helpers + smoke),
and smoke tests for analysis.rdms.sensory."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.spatial.distance import squareform

import analysis.rdms.common as common

# ---------------------------------------------------------------------------
# Small synthetic dataset constants (used to patch module-level values)
# ---------------------------------------------------------------------------
_N = 10
_LEN = _N * (_N - 1) // 2  # 45


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_results(tmp_path, monkeypatch):
    """
    Redirect RESULTS_DIR to a temp dir and shrink expected N so tests don't
    need a real 725-image dataset.
    """
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(common, "_EXPECTED_LEN", _LEN)
    monkeypatch.setattr(common, "_EXPECTED_N", _N)
    return tmp_path


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

def test_load_manifest_shape():
    df = common.load_manifest()
    assert len(df) == 725, f"Expected 725 rows, got {len(df)}"
    assert "curated_path" in df.columns
    assert "category" in df.columns


# ---------------------------------------------------------------------------
# order_hash
# ---------------------------------------------------------------------------

def test_order_hash_stable():
    h1 = common.order_hash()
    h2 = common.order_hash()
    assert h1 == h2


def test_order_hash_length():
    assert len(common.order_hash()) == 16


# ---------------------------------------------------------------------------
# load_image_rgb: alpha compositing
# ---------------------------------------------------------------------------

def test_load_image_rgb_rgba_compositing(tmp_path):
    """Semi-transparent red (alpha=128) over white -> channel values ~(255, 127, 127)."""
    p = tmp_path / "rgba.png"
    img = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    img.save(p)
    arr = common.load_image_rgb(p)
    assert arr.shape == (4, 4, 3)
    assert arr.dtype == np.uint8
    assert arr[0, 0, 0] == 255
    assert 120 <= int(arr[0, 0, 1]) <= 135   # green channel ~127
    assert 120 <= int(arr[0, 0, 2]) <= 135   # blue channel ~127


def test_load_image_rgb_fully_transparent(tmp_path):
    """Fully transparent pixel (alpha=0) -> white (255,255,255)."""
    p = tmp_path / "transparent.png"
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    img.save(p)
    arr = common.load_image_rgb(p)
    np.testing.assert_array_equal(arr, 255)


def test_load_image_rgb_rgb(tmp_path):
    """Plain RGB PNG passes through unchanged."""
    p = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(p)
    arr = common.load_image_rgb(p)
    assert arr.shape == (4, 4, 3)
    assert arr[0, 0, 0] == 10
    assert arr[0, 0, 1] == 20
    assert arr[0, 0, 2] == 30


def test_load_image_rgb_size_mismatch_raises(tmp_path):
    """size parameter raises ValueError when image doesn't match."""
    p = tmp_path / "big.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(p)
    with pytest.raises(ValueError, match="Expected 4x4"):
        common.load_image_rgb(p, size=4)


def test_load_image_rgb_size_ok(tmp_path):
    """size parameter passes silently when image matches."""
    p = tmp_path / "exact.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(p)
    arr = common.load_image_rgb(p, size=4)
    assert arr.shape == (4, 4, 3)


def test_load_image_rgb_background_color_invariant(tmp_path):
    """
    Two RGBA images identical up to background color (alpha=0) must yield
    identical RGB arrays and therefore zero Euclidean distance.

    Rationale: background pixels have alpha=0, so their stored RGB is
    irrelevant. Both composite to white, ensuring sensory distance = 0
    for pixels that are transparent in both images.
    """
    size = 4
    # Top-left 2x2: red foreground (alpha=255) — identical in both images
    # Bottom-right 2x2: background (alpha=0) with different stored RGB
    def make_rgba(bg_rgb: tuple) -> Image.Image:
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        arr[:2, :2] = (255, 0, 0, 255)        # foreground
        arr[2:, 2:] = (*bg_rgb, 0)             # background, alpha=0
        return Image.fromarray(arr, mode="RGBA")

    p_a = tmp_path / "a.png"
    p_b = tmp_path / "b.png"
    make_rgba((0, 0, 0)).save(p_a)         # black stored behind alpha=0
    make_rgba((128, 64, 200)).save(p_b)    # arbitrary color behind alpha=0

    flat_a = common.load_image_rgb(p_a).flatten().astype(np.float64)
    flat_b = common.load_image_rgb(p_b).flatten().astype(np.float64)

    assert np.linalg.norm(flat_a - flat_b) == 0.0


# ---------------------------------------------------------------------------
# save_rdm / load_rdm
# ---------------------------------------------------------------------------

def test_save_creates_npy(patched_results):
    condensed = np.abs(np.random.rand(_LEN))
    common.save_rdm("myrdm", condensed, metric="euclidean", source="tests")
    assert (patched_results / "D_myrdm.npy").exists()


def test_save_load_roundtrip(patched_results):
    condensed = np.random.rand(_LEN)
    common.save_rdm("rt", condensed, metric="cosine", source="tests")
    loaded = common.load_rdm("rt")
    np.testing.assert_array_equal(condensed, loaded)


def test_save_creates_metadata(patched_results):
    condensed = np.random.rand(_LEN)
    common.save_rdm("meta", condensed, metric="km_tree_edge", source="tests",
                    extra={"variant": "pre_shine"})
    records = json.loads((patched_results / "metadata.json").read_text())
    assert any(r["name"] == "meta" for r in records)
    r = next(r for r in records if r["name"] == "meta")
    assert r["metric"] == "km_tree_edge"
    assert r["variant"] == "pre_shine"
    assert "order_hash" in r
    assert "timestamp" in r


def test_save_overwrites_existing_record(patched_results):
    """Re-saving the same name replaces the metadata record (no duplicates)."""
    c1 = np.ones(_LEN)
    c2 = np.zeros(_LEN)
    common.save_rdm("dup", c1, metric="A", source="tests")
    common.save_rdm("dup", c2, metric="B", source="tests")
    records = json.loads((patched_results / "metadata.json").read_text())
    matches = [r for r in records if r["name"] == "dup"]
    assert len(matches) == 1
    assert matches[0]["metric"] == "B"


def test_load_missing_raises(patched_results):
    with pytest.raises(FileNotFoundError):
        common.load_rdm("nonexistent")


def test_load_wrong_length_raises(patched_results):
    bad = np.random.rand(_LEN + 5)
    np.save(patched_results / "D_bad.npy", bad)
    with pytest.raises(ValueError, match="length"):
        common.load_rdm("bad")


def test_load_hash_mismatch_raises(patched_results):
    """Tampered order_hash in metadata -> RuntimeError on load."""
    condensed = np.random.rand(_LEN)
    common.save_rdm("guarded", condensed, metric="euclidean", source="tests")
    # Tamper with stored hash
    meta_path = patched_results / "metadata.json"
    records = json.loads(meta_path.read_text())
    for r in records:
        if r["name"] == "guarded":
            r["order_hash"] = "deadbeef00000000"
    meta_path.write_text(json.dumps(records))
    with pytest.raises(RuntimeError, match="order hash mismatch"):
        common.load_rdm("guarded")


# ---------------------------------------------------------------------------
# squareform round-trip invariants
# ---------------------------------------------------------------------------

def test_squareform_symmetric_zero_diagonal(patched_results):
    """Loaded condensed vector round-trips to symmetric zero-diagonal square matrix."""
    condensed = np.abs(np.random.rand(_LEN))
    common.save_rdm("sq", condensed, metric="euclidean", source="tests")
    loaded = common.load_rdm("sq")
    sq = squareform(loaded)
    assert sq.shape == (_N, _N)
    np.testing.assert_array_almost_equal(sq, sq.T)
    np.testing.assert_array_equal(np.diag(sq), 0.0)


def test_load_rdm_missing_metadata_raises(patched_results):
    """load_rdm raises RuntimeError when metadata.json is absent (not via save_rdm)."""
    condensed = np.abs(np.random.rand(_LEN))
    np.save(patched_results / "D_orphan.npy", condensed)
    # No metadata.json written
    with pytest.raises(RuntimeError, match="metadata.json not found"):
        common.load_rdm("orphan")


def test_load_rdm_missing_record_raises(patched_results):
    """load_rdm raises RuntimeError when metadata.json exists but has no record for name."""
    # Write metadata for a different RDM name
    condensed = np.abs(np.random.rand(_LEN))
    common.save_rdm("other", condensed, metric="euclidean", source="tests")
    # Write the .npy for a name that has no metadata record
    np.save(patched_results / "D_unrecorded.npy", condensed)
    with pytest.raises(RuntimeError, match="No metadata record"):
        common.load_rdm("unrecorded")


# ---------------------------------------------------------------------------
# semantic_km helpers
# ---------------------------------------------------------------------------

import analysis.rdms.semantic_km as km


def test_dir_parts_windows_backslash():
    """Windows backslash paths are normalised; filename is stripped."""
    assert km._dir_parts(r"animate\animal\body\bird\chick1.png") == (
        "animate", "animal", "body", "bird"
    )


def test_dir_parts_forward_slash():
    """Forward-slash paths work the same."""
    assert km._dir_parts("inanimate/natural/food/apple1.png") == (
        "inanimate", "natural", "food"
    )


def test_lca_depth_identical():
    assert km._lca_depth(("a", "b", "c"), ("a", "b", "c")) == 3


def test_lca_depth_no_common():
    assert km._lca_depth(("animate",), ("inanimate",)) == 0


def test_lca_depth_partial():
    assert km._lca_depth(("a", "b", "c"), ("a", "b", "d")) == 2


def test_lca_depth_one_shallower():
    assert km._lca_depth(("a", "b"), ("a", "b", "c")) == 2


# ---------------------------------------------------------------------------
# semantic_km smoke test (build_km_rdm with synthetic manifest)
# ---------------------------------------------------------------------------

def _make_synthetic_manifest(paths: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "curated_path": paths,
        "curated_filename": [p.split("\\")[-1] for p in paths],
        "category": ["X"] * len(paths),
        "source_dataset": ["test"] * len(paths),
        "source_filename_or_url": [""] * len(paths),
        "manual_match_validation": ["V"] * len(paths),
    })


@pytest.fixture()
def patched_km(tmp_path, monkeypatch):
    """Patch km module so build_km_rdm uses a tiny synthetic manifest."""
    paths_4img = [
        r"animate\animal\bird\chick1.png",   # depth 3
        r"animate\animal\bird\chick2.png",   # depth 3, same leaf
        r"animate\animal\fish\fish1.png",    # depth 3, different leaf
        r"inanimate\object\tool\hammer1.png", # depth 3, different top-level
    ]
    synthetic = _make_synthetic_manifest(paths_4img)
    monkeypatch.setattr(km, "load_manifest", lambda: synthetic)
    monkeypatch.setattr(km, "save_rdm", lambda *a, **kw: None)
    return paths_4img


def test_build_km_rdm_same_leaf_distance_1(patched_km):
    """Two images in the same leaf folder must have distance 1 (not 0)."""
    condensed = km.build_km_rdm()
    # pair (0,1): chick1 vs chick2 — same leaf "bird"
    assert condensed[0] == 1.0


def test_build_km_rdm_cross_top_level(patched_km):
    """animate vs inanimate pair has larger distance than within-animate same-leaf pair.

    Condensed vector order for 4 images (0=chick1, 1=chick2, 2=fish1, 3=hammer1):
      index 0 → (0,1) chick1 vs chick2  [same leaf]
      index 1 → (0,2) chick1 vs fish1   [diff leaf, same top]
      index 2 → (0,3) chick1 vs hammer1 [cross top-level]  ← this is what we test
      index 3 → (1,2) chick2 vs fish1   [diff leaf, same top]
    """
    condensed = km.build_km_rdm()
    same_leaf = condensed[0]  # chick1 vs chick2  → distance 1
    cross_top = condensed[2]  # chick1 vs hammer1 → animate vs inanimate
    assert cross_top > same_leaf


def test_build_km_rdm_condensed_length(patched_km):
    condensed = km.build_km_rdm()
    assert len(condensed) == 4 * 3 // 2  # 6 pairs


def test_build_km_rdm_zero_diagonal(patched_km):
    """squareform of result has zero diagonal."""
    condensed = km.build_km_rdm()
    sq = squareform(condensed)
    np.testing.assert_array_equal(np.diag(sq), 0.0)


def test_build_km_rdm_symmetric(patched_km):
    condensed = km.build_km_rdm()
    sq = squareform(condensed)
    np.testing.assert_array_equal(sq, sq.T)


# ---------------------------------------------------------------------------
# sensory smoke test
# ---------------------------------------------------------------------------

import analysis.rdms.sensory as sensory_mod


@pytest.fixture()
def patched_sensory(tmp_path, monkeypatch):
    """Patch sensory module with 4 tiny synthetic images."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    paths = []
    for i in range(4):
        p = img_dir / f"img{i}.png"
        arr = np.full((8, 8, 4), fill_value=[i * 60, 0, 0, 255], dtype=np.uint8)
        Image.fromarray(arr, mode="RGBA").save(p)
        paths.append(p)

    monkeypatch.setattr(sensory_mod, "image_paths", lambda variant: paths)
    monkeypatch.setattr(sensory_mod, "save_rdm", lambda *a, **kw: None)
    return paths


def test_sensory_condensed_length(patched_sensory):
    condensed = sensory_mod.build_sensory_rdm("pre_shine")
    assert len(condensed) == 4 * 3 // 2  # 6 pairs


def test_sensory_non_negative(patched_sensory):
    condensed = sensory_mod.build_sensory_rdm("pre_shine")
    assert np.all(condensed >= 0)


def test_sensory_identical_images_zero_distance(tmp_path, monkeypatch):
    """Two identical images must have zero pairwise distance."""
    p = tmp_path / "same.png"
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(p)
    monkeypatch.setattr(sensory_mod, "image_paths", lambda variant: [p, p])
    monkeypatch.setattr(sensory_mod, "save_rdm", lambda *a, **kw: None)
    condensed = sensory_mod.build_sensory_rdm("pre_shine")
    assert condensed[0] == 0.0
