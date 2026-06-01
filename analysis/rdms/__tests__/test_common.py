"""Tests for analysis.rdms.common."""
from __future__ import annotations

import json

import numpy as np
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
