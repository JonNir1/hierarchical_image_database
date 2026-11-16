"""Tests for analysis.utils.disk_cache: results are reused only under a matching fingerprint."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.utils.disk_cache import cached_array, cached_frame, fingerprint


def test_cached_results_are_reused_only_when_the_fingerprint_matches(tmp_path):
    calls = []

    def array() -> np.ndarray:
        calls.append("array")
        return np.arange(5.0)

    def frame() -> pd.DataFrame:
        calls.append("frame")
        return pd.DataFrame({"x": [1, 2]})

    key = fingerprint(["a", "b"], 100, 0)
    first = cached_array(tmp_path / "a.npz", key, array)
    again = cached_array(tmp_path / "a.npz", key, array)
    stale = cached_array(tmp_path / "a.npz", fingerprint(["a", "b"], 100, 1), array)
    f_first = cached_frame(tmp_path / "f.pkl", key, frame)
    f_again = cached_frame(tmp_path / "f.pkl", key, frame)
    f_stale = cached_frame(tmp_path / "f.pkl", fingerprint("other"), frame)

    assert calls == ["array", "array", "frame", "frame"]  # the matching re-runs were served from disk
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first, stale)
    pd.testing.assert_frame_equal(f_first, f_again)
    pd.testing.assert_frame_equal(f_first, f_stale)
    assert cached_array(None, key, array).shape == (5,)  # path=None disables caching
    assert fingerprint(["a", "b"], 1) != fingerprint(["a", "b"], 2)
