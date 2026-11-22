"""Tests for analysis.s0_build_rdms.common, analysis.s0_build_rdms.semantic.kiani (helpers + smoke),
and smoke tests for analysis.s0_build_rdms.sensory.pixels."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.distance import pdist

from _rdm_helpers import patch_common_results, synthetic_manifest

import analysis.s0_build_rdms.common as common

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
    patch_common_results(monkeypatch, tmp_path, _N)
    return tmp_path


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

@pytest.mark.manual
def test_load_manifest_shape():
    df = common.load_manifest()
    assert len(df) == 725, f"Expected 725 rows, got {len(df)}"
    assert "curated_path" in df.columns
    assert "category" in df.columns


@pytest.mark.manual
def test_load_manifest_paths_are_unique():
    df = common.load_manifest()
    assert df["curated_path"].is_unique, "curated_path must index 725 distinct images"


def test_load_manifest_rejects_duplicate_paths(tmp_path, monkeypatch):
    """A duplicated curated_path is silently destructive: row count is still
    725 and every path still resolves to a real file, so one image would be
    encoded twice and another dropped from every RDM without any downstream
    check noticing. load_manifest() must refuse to return it."""
    broken = tmp_path / "manifest.csv"
    broken.write_text(
        "curated_path,curated_filename,category\n"
        "a\\x1.png,x1.png,HF\n"
        "a\\x1.png,x1.png,HF\n"
        "b\\y1.png,y1.png,HF\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(common, "MANIFEST_PATH", broken)
    common.order_hash.cache_clear()
    with pytest.raises(ValueError, match="duplicated curated_path"):
        common.load_manifest()


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
# image_manifest_index
# ---------------------------------------------------------------------------

def test_image_manifest_index_matches_load_manifest():
    image_to_idx, n_conds = common.image_manifest_index()
    manifest = common.load_manifest()
    assert n_conds == len(manifest)
    canonical = [p.replace("\\", "/") for p in manifest["curated_path"]]
    assert list(image_to_idx.keys()) == canonical
    assert image_to_idx[canonical[0]] == 0
    assert image_to_idx[canonical[-1]] == n_conds - 1


def test_image_manifest_index_normalises_backslashes():
    image_to_idx, _ = common.image_manifest_index()
    assert all("\\" not in path for path in image_to_idx)


def test_image_manifest_index_is_cached():
    first = common.image_manifest_index()
    second = common.image_manifest_index()
    assert first[0] is second[0]  # same dict object, not just equal


# ---------------------------------------------------------------------------
# load_image_rgb
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode, fill, expected_ranges", [
    # semi-transparent red over white -> (255, ~127, ~127)
    pytest.param("RGBA", (255, 0, 0, 128), ((255, 255), (120, 135), (120, 135)), id="rgba_compositing"),
    pytest.param("RGBA", (0, 0, 0, 0), ((255, 255),) * 3, id="fully_transparent_is_white"),
    pytest.param("RGB", (10, 20, 30), ((10, 10), (20, 20), (30, 30)), id="rgb_passthrough"),
])
def test_load_image_rgb_channel_values(tmp_path, mode, fill, expected_ranges):
    p = tmp_path / "img.png"
    Image.new(mode, (4, 4), fill).save(p)
    arr = common.load_image_rgb(p)
    assert arr.shape == (4, 4, 3)
    assert arr.dtype == np.uint8
    for channel, (lo, hi) in enumerate(expected_ranges):
        assert ((arr[..., channel] >= lo) & (arr[..., channel] <= hi)).all()


@pytest.mark.parametrize("side, error", [(8, "Expected 4x4"), (4, None)])
def test_load_image_rgb_size_check(tmp_path, side, error):
    """size=4 raises ValueError for a mismatching image and passes silently for a matching one."""
    p = tmp_path / "img.png"
    Image.new("RGB", (side, side), (0, 0, 0)).save(p)
    if error:
        with pytest.raises(ValueError, match=error):
            common.load_image_rgb(p, size=4)
        return
    assert common.load_image_rgb(p, size=4).shape == (4, 4, 3)


def test_load_image_rgb_background_color_invariant(tmp_path):
    """
    Two RGBA images identical up to background color (alpha=0) must yield
    identical RGB arrays and therefore zero Euclidean distance.

    Rationale: background pixels have alpha=0, so their stored RGB is
    irrelevant. Both composite to white, ensuring sensory distance = 0
    for pixels that are transparent in both images.
    """
    size = 4
    # Top-left 2x2: red foreground (alpha=255): identical in both images
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
# save/load: RDMs (D_*.npy + metadata.json) and embeddings (E_*.npy + embeddings_metadata.json)
# ---------------------------------------------------------------------------

_KINDS = ["rdm", "embeddings"]
_NPY_PREFIX = {"rdm": "D_", "embeddings": "E_"}
_METADATA_FILE = {"rdm": "metadata.json", "embeddings": "embeddings_metadata.json"}
_LOAD = {"rdm": common.load_rdm, "embeddings": common.load_embeddings}
_OVERWRITE_FIELD = {"rdm": "metric", "embeddings": "source"}


def _array_for(kind: str) -> np.ndarray:
    return np.random.rand(_LEN) if kind == "rdm" else np.random.rand(_N, 4)


def _save(kind: str, name: str, field_value: str = "tests") -> None:
    """Save a random array; `field_value` lands in the field that _OVERWRITE_FIELD names."""
    if kind == "rdm":
        common.save_rdm(name, _array_for(kind), metric=field_value, source="tests")
        return
    common.save_embeddings(name, _array_for(kind), source=field_value)


def _records(results_dir, kind: str) -> list[dict]:
    return json.loads((results_dir / _METADATA_FILE[kind]).read_text())


@pytest.mark.parametrize("kind", _KINDS)
def test_save_creates_npy(patched_results, kind):
    _save(kind, "myfile")
    assert (patched_results / f"{_NPY_PREFIX[kind]}myfile.npy").exists()


@pytest.mark.parametrize("kind", _KINDS)
def test_save_overwrites_existing_record(patched_results, kind):
    """Re-saving the same name replaces the metadata record (no duplicates)."""
    field = _OVERWRITE_FIELD[kind]
    _save(kind, "dup", "A")
    _save(kind, "dup", "B")
    matches = [r for r in _records(patched_results, kind) if r["name"] == "dup"]
    assert len(matches) == 1
    assert matches[0][field] == "B"


@pytest.mark.parametrize("kind", _KINDS)
def test_load_missing_file_raises(patched_results, kind):
    with pytest.raises(FileNotFoundError):
        _LOAD[kind]("nonexistent")


@pytest.mark.parametrize("kind", _KINDS)
def test_load_missing_metadata_raises(patched_results, kind):
    """Loading raises RuntimeError when the metadata json is absent (array saved outside save_*)."""
    np.save(patched_results / f"{_NPY_PREFIX[kind]}orphan.npy", _array_for(kind))
    with pytest.raises(RuntimeError, match=f"{_METADATA_FILE[kind]} not found"):
        _LOAD[kind]("orphan")


@pytest.mark.parametrize("kind", _KINDS)
def test_load_missing_record_raises(patched_results, kind):
    """Metadata json exists but has no record for the requested name."""
    _save(kind, "other")
    np.save(patched_results / f"{_NPY_PREFIX[kind]}unrecorded.npy", _array_for(kind))
    with pytest.raises(RuntimeError, match="No metadata record"):
        _LOAD[kind]("unrecorded")


@pytest.mark.parametrize("kind", _KINDS)
def test_load_hash_mismatch_raises(patched_results, kind):
    """Tampered order_hash in the metadata json -> RuntimeError on load."""
    _save(kind, "guarded")
    meta_path = patched_results / _METADATA_FILE[kind]
    records = json.loads(meta_path.read_text())
    for r in records:
        if r["name"] == "guarded":
            r["order_hash"] = "deadbeef00000000"
    meta_path.write_text(json.dumps(records))
    with pytest.raises(RuntimeError, match="order hash mismatch"):
        _LOAD[kind]("guarded")


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


def test_load_wrong_length_raises(patched_results):
    bad = np.random.rand(_LEN + 5)
    np.save(patched_results / "D_bad.npy", bad)
    with pytest.raises(ValueError, match="length"):
        common.load_rdm("bad")


def test_save_load_embeddings_roundtrip(patched_results):
    E = np.random.rand(_N, 8)
    common.save_embeddings("rt", E, source="tests")
    loaded = common.load_embeddings("rt")
    # save_embeddings casts to float32: compare with reduced precision
    np.testing.assert_array_almost_equal(E, loaded, decimal=5)


def test_save_embeddings_creates_metadata(patched_results):
    E = np.random.rand(_N, 8)
    common.save_embeddings("meta", E, source="tests", extra={"variant": "pre_shine"})
    records = json.loads((patched_results / "embeddings_metadata.json").read_text())
    assert any(r["name"] == "meta" for r in records)
    r = next(r for r in records if r["name"] == "meta")
    assert r["embedding_dim"] == 8
    assert r["variant"] == "pre_shine"
    assert "order_hash" in r
    assert "timestamp" in r


def test_save_embeddings_wrong_row_count_raises(patched_results):
    bad = np.random.rand(_N + 1, 4)
    with pytest.raises(ValueError, match="rows"):
        common.save_embeddings("bad", bad, source="tests")

# ---------------------------------------------------------------------------
# euclidean_distances / cosine_distances
# ---------------------------------------------------------------------------

def test_euclidean_distances_matches_pdist(patched_results):
    E = np.random.rand(_N, 5)
    condensed = common.euclidean_distances(E)
    np.testing.assert_array_almost_equal(condensed, pdist(E, metric="euclidean"))


def test_cosine_distances_matches_pdist(patched_results):
    E = np.random.rand(_N, 5)
    condensed = common.cosine_distances(E)
    np.testing.assert_array_almost_equal(condensed, pdist(E, metric="cosine"))


def test_euclidean_distances_identical_rows_zero(patched_results):
    E = np.tile(np.array([1.0, 2.0, 3.0]), (_N, 1))
    condensed = common.euclidean_distances(E)
    np.testing.assert_array_almost_equal(condensed, 0.0)


def test_distances_default_does_not_save(patched_results):
    """save_result defaults to False: no D_*.npy written, no name/source required."""
    E = np.random.rand(_N, 5)
    common.euclidean_distances(E)
    common.cosine_distances(E)
    assert list(patched_results.glob("D_*.npy")) == []


def test_distances_save_result_requires_name_and_source(patched_results):
    E = np.random.rand(_N, 5)
    with pytest.raises(ValueError, match="name and source"):
        common.euclidean_distances(E, save_result=True)


def test_distances_save_result_writes_loadable_rdm(patched_results):
    E = np.random.rand(_N, 5)
    condensed = common.cosine_distances(E, save_result=True, name="emb_rdm", source="tests")
    loaded = common.load_rdm("emb_rdm")
    np.testing.assert_array_equal(condensed, loaded)


def test_distances_save_result_records_metric_and_extra(patched_results):
    E = np.random.rand(_N, 5)
    common.euclidean_distances(
        E, save_result=True, name="emb_rdm2", source="tests",
        extra={"variant": "pre_shine"},
    )
    records = json.loads((patched_results / "metadata.json").read_text())
    r = next(r for r in records if r["name"] == "emb_rdm2")
    assert r["metric"] == "euclidean"
    assert r["variant"] == "pre_shine"


# ---------------------------------------------------------------------------
# kiani helpers
# ---------------------------------------------------------------------------

import analysis.s0_build_rdms.semantic.kiani as km


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


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(("a", "b", "c"), ("a", "b", "c"), 3, id="identical"),
    pytest.param(("animate",), ("inanimate",), 0, id="no_common"),
    pytest.param(("a", "b", "c"), ("a", "b", "d"), 2, id="partial"),
    pytest.param(("a", "b"), ("a", "b", "c"), 2, id="one_shallower"),
])
def test_lca_depth(a, b, expected):
    assert km._lca_depth(a, b) == expected


# ---------------------------------------------------------------------------
# kiani smoke test (build_km_rdm with synthetic manifest)
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_km(tmp_path, monkeypatch):
    """Patch km module so build_km_rdm uses a tiny synthetic manifest."""
    paths_4img = [
        r"animate\animal\bird\chick1.png",   # depth 3
        r"animate\animal\bird\chick2.png",   # depth 3, same leaf
        r"animate\animal\fish\fish1.png",    # depth 3, different leaf
        r"inanimate\object\tool\hammer1.png", # depth 3, different top-level
    ]
    synthetic = synthetic_manifest(paths_4img)
    monkeypatch.setattr(km, "load_manifest", lambda: synthetic)
    monkeypatch.setattr(km, "save_rdm", lambda *a, **kw: None)
    return paths_4img


def test_build_km_rdm_same_leaf_distance_1(patched_km):
    """Two images in the same leaf folder must have distance 1 (not 0)."""
    condensed = km.build_km_rdm()
    # pair (0,1): chick1 vs chick2: same leaf "bird"
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


# ---------------------------------------------------------------------------
# sensory smoke test
# ---------------------------------------------------------------------------

import analysis.s0_build_rdms.sensory.pixels as sensory_mod


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
    monkeypatch.setattr(common, "save_rdm", lambda *a, **kw: None)
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
    monkeypatch.setattr(common, "save_rdm", lambda *a, **kw: None)
    condensed = sensory_mod.build_sensory_rdm("pre_shine")
    assert condensed[0] == 0.0


def test_subset_condensed_reorders_and_keeps_nans():
    from scipy.spatial.distance import squareform

    square = np.arange(16, dtype=float).reshape(4, 4)
    square = (square + square.T) / 2
    np.fill_diagonal(square, 0.0)
    full = squareform(square, checks=False)
    full[0] = np.nan
    idx = np.array([3, 1, 0])
    result = common.subset_condensed(full, idx)
    expected = squareform(squareform(full, checks=False)[np.ix_(idx, idx)], checks=False)
    assert np.array_equal(result, expected, equal_nan=True)
    assert len(result) == 3
    assert np.isnan(result[2]) and not np.isnan(result[:2]).any()
