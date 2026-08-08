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


class TestConfigurationStorage:
    """The optional ``confs.f32`` side-file (MDS configurations, zero-padded to ``max_ndim``)."""

    N_IMAGES, MAX_NDIM, L = 10, 5, 45

    def _store(self, tmp_path, **kw):
        return ResultStore.create(tmp_path / "s", confdist_len=self.L, meta_columns=META_COLS,
                                  n_images=self.N_IMAGES, max_ndim=self.MAX_NDIM, **kw)

    def test_roundtrip_at_max_ndim(self, tmp_path):
        conf = np.random.default_rng(0).random((self.N_IMAGES, self.MAX_NDIM)).astype(np.float32)
        s = self._store(tmp_path)
        s.append(_meta(ndim=self.MAX_NDIM), np.zeros(self.L, np.float32), conf)
        s.close()
        got = ResultStore.open(tmp_path / "s").conf(0, self.MAX_NDIM)
        np.testing.assert_array_equal(got, conf)

    def test_narrower_ndim_is_padded_and_trimmed_back(self, tmp_path):
        """A 3-D fit is stored padded to 5 columns; reading with ndim=3 returns the original."""
        conf = np.random.default_rng(1).random((self.N_IMAGES, 3)).astype(np.float32)
        s = self._store(tmp_path)
        s.append(_meta(ndim=3), np.zeros(self.L, np.float32), conf)
        s.close()
        reopened = ResultStore.open(tmp_path / "s")
        np.testing.assert_array_equal(reopened.conf(0, 3), conf)
        padded = reopened.conf(0)  # no ndim -> full padded width
        assert padded.shape == (self.N_IMAGES, self.MAX_NDIM)
        np.testing.assert_array_equal(padded[:, 3:], 0.0)

    def test_mixed_ndims_stay_row_aligned(self, tmp_path):
        """Rows of differing ndim share one fixed-width file; each must read back independently."""
        rng = np.random.default_rng(2)
        confs = {nd: rng.random((self.N_IMAGES, nd)).astype(np.float32) for nd in (2, 5, 3)}
        s = self._store(tmp_path)
        for rep, (nd, c) in enumerate(confs.items()):
            s.append(_meta(rep=rep, ndim=nd), np.zeros(self.L, np.float32), c)
        s.close()
        reopened = ResultStore.open(tmp_path / "s")
        for row, (nd, c) in enumerate(confs.items()):
            np.testing.assert_array_equal(reopened.conf(row, nd), c)

    def test_failed_run_consumes_no_row_in_either_file(self, tmp_path):
        """A failed MDS run stores neither array, so the two binaries stay in lockstep."""
        conf = np.ones((self.N_IMAGES, self.MAX_NDIM), np.float32)
        s = self._store(tmp_path)
        s.append(_meta(rep=0), np.zeros(self.L, np.float32), conf)
        s.append(_meta(rep=1, status="error"), None)     # failed: no confdist, no conf
        s.append(_meta(rep=2), np.zeros(self.L, np.float32), conf * 2)
        s.close()
        reopened = ResultStore.open(tmp_path / "s")
        assert reopened.metadata()["confdist_row"].tolist() == [0, -1, 1]
        np.testing.assert_array_equal(reopened.conf(1, self.MAX_NDIM), conf * 2)

    def test_conf_required_alongside_confdist(self, tmp_path):
        """Omitting conf would desynchronise the shared row index, so it must raise."""
        s = self._store(tmp_path)
        with pytest.raises(ValueError, match="required alongside"):
            s.append(_meta(), np.zeros(self.L, np.float32))

    @pytest.mark.parametrize("bad_shape", [(9, 5), (10, 6), (10, 0)])
    def test_bad_conf_shape_raises(self, tmp_path, bad_shape):
        s = self._store(tmp_path)
        with pytest.raises(ValueError):
            s.append(_meta(), np.zeros(self.L, np.float32), np.zeros(bad_shape, np.float32))

    def test_create_requires_both_geometry_args(self, tmp_path):
        with pytest.raises(ValueError, match="must be given together"):
            ResultStore.create(tmp_path / "s", confdist_len=self.L,
                               meta_columns=META_COLS, n_images=self.N_IMAGES)

    def test_conf_is_a_small_fraction_of_confdist(self, tmp_path):
        """The whole point of storing coordinates: they are far cheaper than distances."""
        n_images, max_ndim = 725, 10
        L = n_images * (n_images - 1) // 2
        s = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=META_COLS,
                               n_images=n_images, max_ndim=max_ndim)
        s.append(_meta(ndim=max_ndim), np.zeros(L, np.float32),
                 np.zeros((n_images, max_ndim), np.float32))
        s.close()
        conf_bytes = (tmp_path / "s" / "confs.f32").stat().st_size
        confdist_bytes = (tmp_path / "s" / "confdists.f32").stat().st_size
        assert conf_bytes / confdist_bytes < 0.05      # under 5% overhead


class TestBackwardCompatibility:
    """Format-version-1 stores (confdist only) must keep working untouched."""

    def test_v1_store_opens_and_reads(self, tmp_path):
        L = 45
        cd = np.random.default_rng(3).random(L).astype(np.float32)
        s = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=META_COLS)
        s.append(_meta(), cd)
        s.close()
        reopened = ResultStore.open(tmp_path / "s")
        assert not reopened.stores_conf
        np.testing.assert_array_equal(reopened.confdist(0), cd)
        assert not (tmp_path / "s" / "confs.f32").exists()   # no stray side-file

    def test_conf_on_a_v1_store_raises_clearly(self, tmp_path):
        s = ResultStore.create(tmp_path / "s", confdist_len=45, meta_columns=META_COLS)
        s.append(_meta(), np.zeros(45, np.float32))
        s.close()
        reopened = ResultStore.open(tmp_path / "s")
        with pytest.raises(ValueError, match="does not contain configurations"):
            reopened.conf(0)

    def test_passing_conf_to_a_v1_store_raises(self, tmp_path):
        s = ResultStore.create(tmp_path / "s", confdist_len=45, meta_columns=META_COLS)
        with pytest.raises(ValueError, match="without `n_images`"):
            s.append(_meta(), np.zeros(45, np.float32), np.zeros((10, 5), np.float32))


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


# --------------------------------------------------------------------- conf-only stores

class TestConfOnlyStore:
    """A store whose `confdists.f32` was never downloaded must still open and serve `conf`.

    `confdist == pdist(conf)` and a conf row is ~20x smaller, so the cluster analysis pulls confs
    alone (~28 MB against ~500 MB for a full sweep). That download is a supported read mode, not a
    corrupt store, so the record count comes from meta.csv rather than the binary's size.
    """

    def _store(self, tmp_path, n_images=6, max_ndim=4, n_rows=3):
        L = n_images * (n_images - 1) // 2
        rng = np.random.default_rng(0)
        confs = {r: rng.random((n_images, max_ndim)).astype(np.float32) for r in range(n_rows)}
        store = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=META_COLS,
                                   n_images=n_images, max_ndim=max_ndim)
        for r, cf in confs.items():
            store.append(_meta(rep=r), rng.random(L).astype(np.float32), cf)
        store.close()
        return tmp_path / "s", confs

    def _strip_confdists(self, path):
        (path / "confdists.f32").unlink()

    def test_conf_still_readable_without_confdists(self, tmp_path):
        path, confs = self._store(tmp_path)
        self._strip_confdists(path)
        reopened = ResultStore.open(path)
        for r, expected in confs.items():
            np.testing.assert_array_equal(reopened.conf(r), expected)

    def test_record_count_survives(self, tmp_path):
        """The count must come from meta.csv; deriving it from the binary would give 0."""
        path, confs = self._store(tmp_path)
        before = len(ResultStore.open(path))
        self._strip_confdists(path)
        assert len(ResultStore.open(path)) == before == len(confs)

    def test_has_confdists_reports_the_truth(self, tmp_path):
        path, _ = self._store(tmp_path)
        assert ResultStore.open(path).has_confdists
        self._strip_confdists(path)
        assert not ResultStore.open(path).has_confdists

    def test_confdist_falls_back_to_recomputing_from_conf(self, tmp_path):
        """A conf-only store must still serve distances: confdist == pdist(conf) exactly.

        This is the identity that justifies excluding confdists.f32 from every upload, so the
        store should honour it rather than refusing. Refusing is what broke a finished stage-2
        sweep at the metric-tables step, hours after the fits were done.
        """
        from scipy.spatial.distance import pdist

        path, confs = self._store(tmp_path)
        self._strip_confdists(path)
        got = ResultStore.open(path).confdist(0)
        np.testing.assert_allclose(got, pdist(confs[0]), atol=1e-6)

    def test_a_truncated_confdists_file_serves_the_missing_rows_from_conf(self, tmp_path):
        """The resume case: the file restarts at zero while meta.csv keeps counting up.

        Restoring a store without confdists.f32 and then appending leaves recorded row indices
        ahead of the file's contents. Sizing the memmap from meta.csv then raised
        "mmap length is greater than file size"; sizing it from the file and recomputing the
        overhang is correct. Note this fixture stores UNRELATED random confdists, so a row still
        present in the file must come back unchanged while only the overhang is recomputed - which
        is exactly the boundary worth pinning.
        """
        from scipy.spatial.distance import pdist

        path, confs = self._store(tmp_path)
        store = ResultStore.open(path)
        n_pairs = store.confdist_len
        stored_row0 = store.confdist(0).copy()

        blob = (path / "confdists.f32").read_bytes()
        (path / "confdists.f32").write_bytes(blob[:n_pairs * 4])   # keep only row 0

        reopened = ResultStore.open(path)
        assert reopened._confdists_on_disk() == 1
        assert len(reopened) == len(confs)
        # Row 0 is still on disk, so it is served verbatim rather than recomputed.
        np.testing.assert_array_equal(reopened.confdist(0), stored_row0)
        # Rows past the file are recomputed from their configuration.
        for row in (1, 2):
            np.testing.assert_allclose(reopened.confdist(row), pdist(confs[row]), atol=1e-6)

    def test_an_out_of_range_row_still_raises(self, tmp_path):
        path, confs = self._store(tmp_path)
        self._strip_confdists(path)
        with pytest.raises(IndexError):
            ResultStore.open(path).confdist(len(confs))

    def test_conf_trimming_still_works(self, tmp_path):
        path, confs = self._store(tmp_path, max_ndim=4)
        self._strip_confdists(path)
        trimmed = ResultStore.open(path).conf(0, ndim=2)
        assert trimmed.shape == (6, 2)
        np.testing.assert_array_equal(trimmed, confs[0][:, :2])

    def test_iter_results_yields_none_rather_than_raising(self, tmp_path):
        path, confs = self._store(tmp_path)
        self._strip_confdists(path)
        out = list(ResultStore.open(path).iter_results())
        assert len(out) == len(confs)
        assert all(cd is None for _, cd in out)

    def test_a_normal_store_is_unaffected(self, tmp_path):
        """The meta.csv-derived count must agree with the old file-size derivation."""
        path, confs = self._store(tmp_path)
        reopened = ResultStore.open(path)
        assert reopened.has_confdists and len(reopened) == len(confs)
        assert reopened.confdist(0).shape == (15,)
        np.testing.assert_array_equal(reopened.conf(1), confs[1])

    def test_failed_rows_do_not_inflate_the_binary_row_count(self, tmp_path):
        """Metadata-only records carry confdist_row = -1 and occupy no row in either binary.

        `len(store)` counts metadata records (so it stays 2 here); the binary row count is what
        must ignore the -1, and reading past it must raise rather than return padding.
        """
        L = 15
        store = ResultStore.create(tmp_path / "f", confdist_len=L, meta_columns=META_COLS,
                                   n_images=6, max_ndim=4)
        rng = np.random.default_rng(1)
        store.append(_meta(rep=0), rng.random(L).astype(np.float32),
                     rng.random((6, 4)).astype(np.float32))
        store.append(_meta(rep=1, status="error"))          # no confdist, no conf
        store.close()

        reopened = ResultStore.open(tmp_path / "f")
        assert len(reopened) == 2, "both metadata records are still records"
        assert reopened.conf(0).shape == (6, 4)
        with pytest.raises(IndexError):
            reopened.conf(1)
