"""Tests for analysis.behavioral.generalizability's data-generation functions.

Uses a small synthetic multi-participant SpAM dataset covering every status this
module's exclusion logic must handle (mirroring analysis/rdms/__tests__/test_behavioral.py's
JSON/DataFrame-level synthetic-fixture pattern, since -- unlike test_reliability.py --
these functions operate on df_participants['status'] directly, not just pre-parsed
trial tuples): retained (status='full data', task_version>=4.0), OOS-eligible
('screened out', 'false_positive', pilot-era 'full data' with task_version<4.0), and
hard-excluded ('revoked consent', 'missing data') -- so the exclusion logic itself is
directly exercised, not just the happy path.

Scoped to generalizability.py's data-generation functions only (_oos_participant_ids,
build_oos_rdm, compare_generalizability, run_generalizability_comparison) -- the
R-dependent MDS functions (compare_generalizability_mds, run_generalizability_mds_comparison)
need a live R installation and are verified via the real notebook execution instead,
consistent with mds_rsatoolbox_compare.py/mds_deep_dive.py not being unit-tested anywhere
in this repo.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import analysis.behavioral.generalizability as generalizability

# 6 synthetic images -> C(6,2) = 15 pairs
_IMAGES = ["img0.png", "img1.png", "img2.png", "img3.png", "img4.png", "img5.png"]


def _pairwise_json(pairs: dict[tuple[str, str], float], variant_dir: str) -> str:
    items = [
        {"src1": f"./images/{variant_dir}/{a}", "src2": f"./images/{variant_dir}/{b}", "distance": d}
        for (a, b), d in pairs.items()
    ]
    return json.dumps(items)


def _full_pairs(base: float, rng: np.random.Generator) -> dict[tuple[str, str], float]:
    pairs = {}
    for i in range(len(_IMAGES)):
        for j in range(i + 1, len(_IMAGES)):
            pairs[(_IMAGES[i], _IMAGES[j])] = float(base + rng.uniform(0, 0.02))
    return pairs


def _synthetic_manifest() -> pd.DataFrame:
    return pd.DataFrame({"curated_path": _IMAGES})


def _synthetic_participants() -> pd.DataFrame:
    return pd.DataFrame([
        # retained (analytical cohort)
        {"participant_id": "p1", "status": "full data", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "p2", "status": "full data", "shine_variant": "post", "task_version": 4.0},
        # OOS-eligible: pre
        {"participant_id": "q1", "status": "screened out", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "q2", "status": "false_positive", "shine_variant": "pre", "task_version": 4.01},
        {"participant_id": "q3", "status": "full data", "shine_variant": "pre", "task_version": 1.0},  # pilot
        # OOS-eligible: post
        {"participant_id": "q4", "status": "screened out", "shine_variant": "post", "task_version": 4.0},
        {"participant_id": "q5", "status": "false_positive", "shine_variant": "post", "task_version": 4.01},
        {"participant_id": "q6", "status": "full data", "shine_variant": "post", "task_version": 2.0},  # pilot
        # hard-excluded regardless of purpose -- must never appear in OOS, even though
        # x1 has real trial data (excluded by status, not by absence of data)
        {"participant_id": "x1", "status": "revoked consent", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "x2", "status": "missing data", "shine_variant": "post", "task_version": np.nan},
    ])


def _synthetic_trials() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for pid, variant_dir, base in [
        ("p1", "pre_shine", 0.10), ("q1", "pre_shine", 0.12), ("q2", "pre_shine", 0.14),
        ("q3", "pre_shine", 0.16), ("x1", "pre_shine", 0.99),
        ("p2", "post_shine", 0.20), ("q4", "post_shine", 0.22), ("q5", "post_shine", 0.24),
        ("q6", "post_shine", 0.26),
    ]:
        rows.append({
            "participant_id": pid, "is_catch": False, "block_type": "experimental",
            "pairwise_distances": _pairwise_json(_full_pairs(base, rng), variant_dir),
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def patched_generalizability(monkeypatch):
    """Monkeypatches both generalizability.py's own load_manifest/load_data/save_rdm
    (used by build_oos_rdm) AND analysis.rdms.behavioral's (used internally by
    build_behavioral_rdm, imported into generalizability.py and called as-is)."""
    import analysis.rdms.behavioral as behavioral

    fake_data = {"participants": _synthetic_participants(), "trials": _synthetic_trials()}
    saved: dict = {}

    def fake_save_rdm(name, condensed, *, metric, source, extra=None):
        saved[name] = {"condensed": condensed, "metric": metric, "source": source, "extra": extra or {}}
        return None

    for module in (generalizability, behavioral):
        monkeypatch.setattr(module, "load_manifest", _synthetic_manifest)
        monkeypatch.setattr(module, "load_data", lambda cache_dir, prod_only=False: fake_data)
        monkeypatch.setattr(module, "save_rdm", fake_save_rdm)

    return saved


# ---------------------------------------------------------------------------
# _oos_participant_ids
# ---------------------------------------------------------------------------

def test_oos_participant_ids_pre_shine(patched_generalizability):
    oos = generalizability._oos_participant_ids(_synthetic_participants(), "pre_shine")
    assert set(oos["participant_id"]) == {"q1", "q2", "q3"}


def test_oos_participant_ids_post_shine(patched_generalizability):
    oos = generalizability._oos_participant_ids(_synthetic_participants(), "post_shine")
    assert set(oos["participant_id"]) == {"q4", "q5", "q6"}


def test_oos_participant_ids_excludes_retained_and_hard_excluded(patched_generalizability):
    """p1 (retained), x1 (revoked consent, despite having real trial data) must never
    appear, for either variant."""
    oos_pre = generalizability._oos_participant_ids(_synthetic_participants(), "pre_shine")
    oos_post = generalizability._oos_participant_ids(_synthetic_participants(), "post_shine")
    all_oos_ids = set(oos_pre["participant_id"]) | set(oos_post["participant_id"])
    assert "p1" not in all_oos_ids
    assert "p2" not in all_oos_ids
    assert "x1" not in all_oos_ids
    assert "x2" not in all_oos_ids


# ---------------------------------------------------------------------------
# build_oos_rdm
# ---------------------------------------------------------------------------

def test_build_oos_rdm_uses_only_oos_participants(patched_generalizability):
    condensed, info = generalizability.build_oos_rdm("pre_shine", "mean", cache_dir="unused")
    assert info["n_oos_participants"] == 3
    assert info["oos_status_breakdown"] == {"screened out": 1, "false_positive": 1, "full data": 1}
    assert len(condensed) == len(_IMAGES) * (len(_IMAGES) - 1) // 2


def test_build_oos_rdm_saves_with_oos_suffixed_name(patched_generalizability):
    generalizability.build_oos_rdm("post_shine", "simple", cache_dir="unused", save=True)
    assert "behav_post_simple_oos" in patched_generalizability


def test_build_oos_rdm_save_false_does_not_save(patched_generalizability):
    generalizability.build_oos_rdm("post_shine", "simple", cache_dir="unused", save=False)
    assert "behav_post_simple_oos" not in patched_generalizability


# ---------------------------------------------------------------------------
# compare_generalizability / run_generalizability_comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aggregation", ["mean", "simple", "evidence"])
def test_compare_generalizability_returns_merged_dict(patched_generalizability, aggregation):
    result = generalizability.compare_generalizability(
        "pre_shine", aggregation, cache_dir="unused", min_overlap=3, save_oos=False,
    )
    assert result["cohort"] == "pre"
    assert result["aggregation"] == aggregation
    assert result["n_oos_participants"] == 3
    assert not result["underpowered"]
    assert -1.0 <= result["kendall"] <= 1.0
    assert -1.0 <= result["spearman"] <= 1.0


def test_run_generalizability_comparison_shape(patched_generalizability, monkeypatch):
    # run_generalizability_comparison() hardcodes _DEFAULT_MIN_OVERLAP=20 unless overridden;
    # this synthetic fixture only has 15 possible pairs, so pass a small min_overlap.
    df = generalizability.run_generalizability_comparison(cache_dir="unused", min_overlap=3, save_oos=False)
    # 2 cohorts x 3 aggregations x 2 metrics = 12 rows
    assert len(df) == 12
    assert set(df["cohort"]) == {"pre", "post"}
    assert set(df["aggregation"]) == {"mean", "simple", "evidence"}
    assert set(df["metric"]) == {"kendall", "spearman"}
    assert not df["underpowered"].any()
