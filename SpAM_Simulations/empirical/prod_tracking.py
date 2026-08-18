"""How the deployed study is tracking against what the simulation predicted.

The v5 report validates its generative model against 41 pilot subjects on task versions 1.0-3.06.
The deployed task is v4.0, with a screening block the pilot never had. This module scores the live
cohort on the same observables, so the report's claims can be checked against the study they were
written to plan.

**What may be computed here, and what may not.** Collection is still running, so this module is
deliberately restricted to two tiers:

* *participant behaviour* - reliability, agreement, canvas usage, screening outcomes. Properties of
  people rather than of the images, and none is a test statistic for any registered hypothesis.
* *measurement coverage* - pair coverage, graph connectivity, per-level pair counts, allocation
  balance, the overall raw noise ceiling. Counts and reliabilities, not effects.

Nothing here builds an embedding, and nothing here computes a mean rating per semantic level: the
first is the deliverable and the second previews RQ1b. The dividing line is that **counts and
reliabilities are safe; means of the ratings themselves, broken down by semantic level, are not.**

**Where the cohorts are pooled and where they are split.** ``docs/WORKFLOW.md`` already sets the
principle for the noise model: it estimates a property of subjects rather than of the stimulus set,
so it does not filter by SHINE variant. Extending that, everything describing *participants* is
pooled (with the per-variant split reported beside it as a declared manipulation check), and
everything describing *the measurement of the image set* is per cohort, because each cohort gets its
own RDM and its own connectivity stopping rule. Pooled coverage is a number the analysis never uses.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from SpAM_Simulations.empirical import gt_construction as gtc
from SpAM_Simulations.empirical import gt_diagnostics as gtd
from SpAM_Simulations.empirical import screening_audit as sa
from SpAM_Simulations.empirical import subjects as subj
from SpAM_Simulations.measures import validity
from SpAM_Simulations.models import block_design

POOLED = "both cohorts"
VARIANT_ORDER = ("pre", "post")


def _stamp(subjects: Sequence) -> Dict[str, object]:
    """Provenance every table carries, because the snapshot ages as collection continues."""
    per_variant = pd.Series([s.shine_variant for s in subjects]).value_counts().to_dict()
    # Deliberately prefixed. Several tables carry their own per-cohort `n_subjects`, and an
    # unprefixed key here would silently overwrite it through `.assign`.
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_n_subjects": len(subjects),
        "snapshot_n_per_variant": "; ".join(f"{k}={v}" for k, v in sorted(per_variant.items())),
    }


def _with_stamp(frame: pd.DataFrame, subjects: Sequence) -> pd.DataFrame:
    return frame.assign(**_stamp(subjects))


def by_variant(subjects: Sequence) -> List[tuple]:
    """``(label, cohort)`` for the pooled set and then each variant, in a fixed order."""
    out = [(POOLED, list(subjects))]
    for variant in VARIANT_ORDER:
        cohort = [s for s in subjects if s.shine_variant == variant]
        if cohort:
            out.append((variant, cohort))
    return out


# --------------------------------------------------------------------------- participant behaviour
def reliability(subjects: Sequence) -> pd.DataFrame:
    """Per-subject within-subject test-retest, with the variant kept for the split.

    One row per subject rather than a summary, because the deciding figure is a distribution: the
    pre-registration excludes the lowest decile *per cohort*, which cannot be read off a mean.
    """
    values = subj.subject_reliability_sample(subjects)
    frame = pd.DataFrame({
        "participant_id": [s.participant_id for s in subjects],
        "shine_variant": [s.shine_variant for s in subjects],
        "task_version": [s.task_version for s in subjects],
        "test_retest": values,
    })
    return _with_stamp(frame, subjects)


def agreement(subjects: Sequence, min_overlap: int = 20) -> pd.DataFrame:
    """Mean pairwise between-subject Spearman, pooled and per variant."""
    rows = []
    for label, cohort in by_variant(subjects):
        if len(cohort) < 2:
            continue
        out = subj.between_subject_agreement(subj.stack_distances(cohort), min_overlap=min_overlap)
        rows.append({"group": label, "n_subjects": len(cohort), **out})
    return _with_stamp(pd.DataFrame(rows), subjects)


def noise_curve(subjects: Sequence, n_bins: int = 10) -> pd.DataFrame:
    """Binned RMSE between the two placements of a repeated pair, in canvas-diagonal units.

    ``rescale="none"``: the deployed task divides every distance by the arena diagonal, and under
    task-v5 so does the simulator, so all three sources already share one [0, 1] scale and the
    median rescaling the pilot comparison needed is not wanted here.
    """
    rows = []
    for label, cohort in by_variant(subjects):
        pairs = validity.repeat_pairs(cohort)
        if pairs[0].size == 0:
            continue
        table = validity.noise_vs_distance(*pairs, n_bins=n_bins, rescale="none")
        rows.append(table.assign(group=label, n_subjects=len(cohort)))
    if not rows:
        return _with_stamp(pd.DataFrame(), subjects)
    return _with_stamp(pd.concat(rows, ignore_index=True), subjects)


def noise_curve_shape(curve: pd.DataFrame) -> pd.DataFrame:
    """The inverted-U summary per group, which is the quantity the model is scored on."""
    rows = []
    for label, sub in curve.groupby("group", sort=False):
        rows.append({"group": label, **validity.noise_curve_shape(sub)})
    return pd.DataFrame(rows)


def null_distance(subjects: Sequence, num_dots: int = 20, num_trials: int = 2000,
                  seed: int = 42) -> pd.DataFrame:
    """Observed pairwise distances against uniformly-random placement on the same canvas.

    The assumption-free floor: if participants were dropping images without regard to similarity
    their distances would look like the null. The simulation fails this check in a specific
    direction (it over-disperses), which is why it is worth running on real data.
    """
    cohorts = {label: c for label, c in by_variant(subjects)}
    pre = cohorts.get("pre", [])
    post = cohorts.get("post", [])
    if not pre or not post:
        # Only one variant present: compare the pooled cohort against itself-free labels.
        pooled = np.concatenate(validity.repeat_pairs(cohorts[POOLED])) if cohorts[POOLED] else \
            np.array([])
        table = validity.null_distance_summary(pooled, pooled, num_dots=num_dots,
                                               num_trials=num_trials, seed=seed,
                                               labels=(POOLED, POOLED))
        table = table.drop_duplicates(subset="source")
    else:
        table = validity.null_distance_summary(
            np.concatenate(validity.repeat_pairs(pre)),
            np.concatenate(validity.repeat_pairs(post)),
            num_dots=num_dots, num_trials=num_trials, seed=seed,
            labels=("pre-SHINE participants", "post-SHINE participants"))
    return _with_stamp(table, subjects)


# --------------------------------------------------------------------------- screening outcomes
def screening_outcomes(data_dir: str, manifest_path: str,
                       config_path: str = sa.DEFAULT_CONFIG) -> pd.DataFrame:
    """Early fails, false positives, and clean passes, with per-criterion attribution.

    Three outcomes, only two of which the deployed task can see:

    * **early fail** - the gate rejected them in-task, on the screening block;
    * **false positive** - they cleared the gate, completed at full rate, and then failed the same
      rule on their experimental block;
    * **clean pass** - cleared the gate and held up.

    The simulation produces a counterpart only for the reliability criterion
    (:data:`screening_audit.SIMULABLE`), so the attribution columns are what make the comparison
    honest rather than decorative.
    """
    from analysis.utils.parser import load_data

    thresholds = sa.load_thresholds(config_path)
    data = load_data(data_dir)
    participants = data["participants"]
    trials = data["trials"]
    prod = participants[participants["cohort"] == "production"]
    # Only people who actually sat the task can pass or fail its gate. "revoked consent" and
    # "missing data" are recruitment attrition, a different quantity, and counting them as clean
    # passes would flatter the pass rate.
    attempted = (prod[prod["status"].isin(["full data", "screened out"])]
                 if "status" in prod.columns else prod)
    n_not_attempted = len(prod) - len(attempted)

    rows = []
    for _, participant in attempted.iterrows():
        pid = participant["participant_id"]
        mine = trials[(trials["participant_id"] == pid) & (~trials["is_catch"].astype(bool))]
        status = participant.get("status", "")
        experimental = mine[mine.get("block_type", "experimental") == "experimental"] \
            if "block_type" in mine.columns else mine
        outcome, reasons = "clean pass", []
        if status == "screened out":
            outcome = "early fail"
            reasons = _reasons_from_participant(participant)
        elif len(experimental):
            audit = sa.evaluate_screening(experimental, thresholds)
            if not audit["pass"]:
                outcome, reasons = "false positive", audit["reasons"]
        criteria = sa.criteria_of(reasons)
        rows.append({
            "participant_id": pid,
            "shine_variant": participant.get("shine_variant", ""),
            "outcome": outcome,
            "n_experimental_trials": int(len(experimental)),
            "failed_reliability": "reliability" in criteria,
            "failed_move_ratio": "move_ratio" in criteria,
            "failed_distance_sd": "distance_sd" in criteria,
            "reasons": " | ".join(reasons),
        })
    frame = pd.DataFrame(rows)
    frame["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame["n_not_attempted"] = n_not_attempted
    return frame


def _reasons_from_participant(participant: pd.Series) -> List[str]:
    """The screening_eval row's own reasons, as the browser recorded them."""
    raw = participant.get("reasons")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, str):
        import json
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        return list(parsed) if isinstance(parsed, list) else [str(parsed)]
    return list(raw)


def screening_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Outcome shares, pooled and per variant, plus which criteria drove each failure."""
    rows = []
    groups = [(POOLED, outcomes)]
    for variant in VARIANT_ORDER:
        sub = outcomes[outcomes["shine_variant"] == variant]
        if len(sub):
            groups.append((variant, sub))
    for label, sub in groups:
        n = len(sub)
        counts = sub["outcome"].value_counts().to_dict()
        rows.append({
            "group": label, "n_candidates": n,
            "clean_pass": counts.get("clean pass", 0),
            "early_fail": counts.get("early fail", 0),
            "false_positive": counts.get("false positive", 0),
            "pass_rate": (n - counts.get("early fail", 0)) / n if n else np.nan,
            "false_positive_rate": counts.get("false positive", 0) /
                                   max(n - counts.get("early fail", 0), 1),
            "failed_reliability": int(sub["failed_reliability"].sum()),
            "failed_move_ratio": int(sub["failed_move_ratio"].sum()),
            "failed_distance_sd": int(sub["failed_distance_sd"].sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- measurement coverage
def coverage(subjects: Sequence) -> pd.DataFrame:
    """Pair coverage and observed-graph connectivity, **per cohort**.

    Per cohort and not pooled, deliberately: each SHINE cohort is embedded separately, so the
    pooled figure describes a matrix nothing in the analysis plan ever builds. Connectivity is the
    pre-registration's own stopping criterion.
    """
    rows = []
    for variant in VARIANT_ORDER:
        cohort = [s for s in subjects if s.shine_variant == variant]
        if not cohort:
            continue
        _, weights = gtc.aggregate_subjects(cohort)
        rows.append({
            "cohort": variant, "n_subjects": len(cohort),
            "pair_coverage": float(gtc.coverage_of(cohort)),
            "n_components": int(gtc.n_components(weights)),
            "connected": bool(gtc.is_connected(cohort)),
        })
    return _with_stamp(pd.DataFrame(rows), subjects)


def level_coverage(subjects: Sequence, manifest_images: Sequence[str]) -> pd.DataFrame:
    """How many pairs at each semantic level anyone has judged yet, per cohort.

    Counts only. The mean *distance* per level is the RQ1b preview and is deliberately absent.
    """
    levels = validity.hierarchy_levels(manifest_images)
    rows = []
    for variant in VARIANT_ORDER:
        cohort = [s for s in subjects if s.shine_variant == variant]
        if not cohort:
            continue
        table = gtd.level_coverage(cohort, levels)
        rows.append(table.assign(cohort=variant, n_subjects=len(cohort)))
    if not rows:
        return _with_stamp(pd.DataFrame(), subjects)
    return _with_stamp(pd.concat(rows, ignore_index=True), subjects)


def noise_ceiling(subjects: Sequence, manifest_images: Sequence[str],
                  n_splits: int = 20, seed: int = 0) -> pd.DataFrame:
    """Split-half reliability of the pooled ratings, per cohort, **overall only**.

    ``gt_diagnostics.raw_noise_ceiling`` reports this per semantic level; only the overall row is
    kept, because the per-level breakdown is a preview of the registered level-stratified analysis.
    """
    levels = validity.hierarchy_levels(manifest_images)
    rng = np.random.default_rng(seed)
    rows = []
    for variant in VARIANT_ORDER:
        cohort = [s for s in subjects if s.shine_variant == variant]
        if len(cohort) < 4:
            continue
        # All pairs collapsed into one pseudo-level: this is an overall figure, and the per-level
        # breakdown raw_noise_ceiling would otherwise give is a preview of the registered
        # level-stratified analysis.
        table = gtd.raw_noise_ceiling(cohort, np.zeros_like(levels), n_splits=n_splits, rng=rng)
        table = table.assign(level_name="all pairs").drop(columns=["level"])
        rows.append(table.assign(cohort=variant, n_subjects=len(cohort)))
    if not rows:
        return _with_stamp(pd.DataFrame(), subjects)
    return _with_stamp(pd.concat(rows, ignore_index=True), subjects)


def allocation(data_dir: str, manifest_path: str) -> pd.DataFrame:
    """Balance of the allocation the study has actually produced, per cohort.

    Scored from the observed trials, which is the only way to compare the deployed random
    assignment against the simulated arms on equal terms. A ``Subject`` keeps condensed per-pair
    counts and so cannot recover which images shared a trial; the raw ``pairwise_distances`` column
    can, since every image in a trial pairs with every other.

    Catch trials are excluded: they draw from a separate pictogram pool and are not part of the
    allocation being evaluated.
    """
    from analysis.utils.parser import load_data, parse_pairwise_distances

    _, rel2idx = subj.load_manifest(manifest_path)
    data = load_data(data_dir)
    participants = data["participants"]
    trials = data["trials"]
    prod = participants[participants["cohort"] == "production"]
    variant_of = dict(zip(prod["participant_id"], prod.get("shine_variant", "")))
    n_images = len(rel2idx)

    per_cohort: Dict[str, List[List[int]]] = {v: [] for v in VARIANT_ORDER}
    for _, row in trials.iterrows():
        pid = row["participant_id"]
        if pid not in variant_of:
            continue
        if bool(row.get("is_catch", False)):
            continue
        indices = set()
        for a, b in parse_pairwise_distances(row["pairwise_distances"]):
            for src in (a, b):
                idx = rel2idx.get(subj._src_to_relpath(src))
                if idx is not None:
                    indices.add(idx)
        if indices:
            per_cohort.setdefault(variant_of[pid], []).append(sorted(indices))

    rows = []
    for variant in VARIANT_ORDER:
        blocks = per_cohort.get(variant, [])
        if not blocks:
            continue
        widths = {len(b) for b in blocks}
        if len(widths) > 1:
            # design_stats wants a rectangular (n_blocks, k). Trials that lost images to a
            # malformed distances payload are dropped rather than padded, since padding would
            # invent co-occurrences that never happened.
            k = max(widths)
            blocks = [b for b in blocks if len(b) == k]
        arr = np.asarray(blocks, dtype=np.int64)
        rows.append({"cohort": variant, "n_trials": int(arr.shape[0]),
                     "images_per_trial": int(arr.shape[1]),
                     **block_design.design_stats(arr, n_images)})
    frame = pd.DataFrame(rows)
    frame["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return frame


def track(subjects: Sequence, manifest_images: Sequence[str], data_dir: str,
          manifest_path: str) -> Dict[str, pd.DataFrame]:
    """Every table, keyed by the filename stem the report reads."""
    curve = noise_curve(subjects)
    outcomes = screening_outcomes(data_dir, manifest_path)
    tables = {
        "prod_reliability": reliability(subjects),
        "prod_agreement": agreement(subjects),
        "prod_noise_curve": curve,
        "prod_null_distance": null_distance(subjects),
        "prod_coverage": coverage(subjects),
        "prod_level_coverage": level_coverage(subjects, manifest_images),
        "prod_noise_ceiling": noise_ceiling(subjects, manifest_images),
        "prod_allocation": allocation(data_dir, manifest_path),
        "prod_screening": outcomes,
        "prod_screening_summary": screening_summary(outcomes),
    }
    if not curve.empty:
        tables["prod_noise_curve_shape"] = noise_curve_shape(curve)
    return tables
