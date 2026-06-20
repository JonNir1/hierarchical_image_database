"""Tests for the compact streamable ResultStore."""
import numpy as np
import pytest

from SpAM_Simulations.storage import ResultStore

META_COLS = ["num_subjects", "trials_per_subject", "rep", "ndim", "niter", "stress", "status"]


def _meta(**kw):
    base = dict(num_subjects=10, trials_per_subject=8, rep=0, ndim=5,
                niter=42, stress=0.1, status="success")
    base.update(kw)
    return base


def test_roundtrip_lossless_float32(tmp_path):
    L = 45  # N=10 condensed length
    rng = np.random.default_rng(0)
    confdists = {r: rng.random(L).astype(np.float32) for r in range(4)}
    store = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=META_COLS)
    for r, cd in confdists.items():
        store.append(_meta(rep=r), cd)
    store.close()

    reopened = ResultStore.open(tmp_path / "s")
    assert len(reopened) == 4
    for meta, cd in reopened.iter_results():
        # float32 storage must reconstruct the values exactly (lossless at float32)
        np.testing.assert_array_equal(cd, confdists[meta["rep"]])


def test_float64_input_downcast_is_exact_at_float32(tmp_path):
    L = 45
    cd64 = np.random.default_rng(1).random(L)  # float64
    store = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=META_COLS)
    store.append(_meta(), cd64)
    store.close()
    got = ResultStore.open(tmp_path / "s").confdist(0)
    np.testing.assert_array_equal(got, cd64.astype(np.float32))


def test_missing_confdist_stored_as_none(tmp_path):
    L = 45
    store = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=META_COLS)
    store.append(_meta(rep=0, status="success"), np.zeros(L, np.float32))
    store.append(_meta(rep=1, status="max_iters"), None)  # failed run: no confdist
    store.close()

    reopened = ResultStore.open(tmp_path / "s")
    results = list(reopened.iter_results())
    assert results[0][1] is not None
    assert results[1][1] is None
    df = reopened.metadata()
    assert list(df["status"]) == ["success", "max_iters"]
    assert df["confdist_row"].tolist() == [0, -1]


def test_append_reopen_append(tmp_path):
    L = 10
    s = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=META_COLS)
    s.append(_meta(rep=0), np.full(L, 1.0, np.float32))
    s.close()
    s2 = ResultStore.open(tmp_path / "s")  # resume
    s2.append(_meta(rep=1), np.full(L, 2.0, np.float32))
    s2.close()
    s3 = ResultStore.open(tmp_path / "s")
    assert len(s3) == 2
    np.testing.assert_array_equal(s3.confdist(0), np.full(L, 1.0, np.float32))
    np.testing.assert_array_equal(s3.confdist(1), np.full(L, 2.0, np.float32))


def test_wrong_confdist_length_raises(tmp_path):
    store = ResultStore.create(tmp_path / "s", confdist_len=45, meta_columns=META_COLS)
    with pytest.raises(ValueError):
        store.append(_meta(), np.zeros(10, np.float32))


def test_missing_meta_column_raises(tmp_path):
    store = ResultStore.create(tmp_path / "s", confdist_len=45, meta_columns=META_COLS)
    with pytest.raises(ValueError):
        store.append({"num_subjects": 10}, np.zeros(45, np.float32))


def test_create_no_overwrite(tmp_path):
    ResultStore.create(tmp_path / "s", confdist_len=45, meta_columns=META_COLS)
    with pytest.raises(FileExistsError):
        ResultStore.create(tmp_path / "s", confdist_len=45, meta_columns=META_COLS)
    ResultStore.create(tmp_path / "s", confdist_len=45, meta_columns=META_COLS, overwrite=True)


def test_storage_smaller_than_float64(tmp_path):
    # float32 binary store should be ~half the heavy-array footprint of float64
    L = 1000
    n = 50
    rng = np.random.default_rng(2)
    store = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=META_COLS)
    for r in range(n):
        store.append(_meta(rep=r), rng.random(L))
    store.close()
    f32_bytes = (tmp_path / "s" / "confdists.f32").stat().st_size
    assert f32_bytes == n * L * 4  # exact float32 packing
    assert f32_bytes < n * L * 8  # smaller than float64
