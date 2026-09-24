"""
TOST equivalence of the pre- and post-SHINE cohorts' agreement with the external Kriegeskorte (2008b)
/ Mur (2013) RDMs, on the 42 visually approved overlap images (Q1, analysis 2).

Effect: D = tau(pre combined RDM, target) - tau(post combined RDM, target), per external target
(Mur 2013 judgments primary; human and monkey IT secondary) and per statistic (Kendall tau-a
primary, Spearman secondary). The cohorts are disjoint participants, so each is bootstrapped on
its own (participants resampled, `simple` aggregation re-run, restricted to the overlap images) and
D's (1 - 2*alpha) percentile CI must lie inside +/- bound, with the project's bound rule
(analysis.utils.equivalence.equivalence_test_bootstrap_effect: smallest bound with 80% power at a
true D = 0, rounded up to 0.01). Two pair sets are scored on the same resamples: each cohort's own
observed pairs (as in run_k92) and only the pairs both cohorts observed in the full data.

--validate applies the identical procedure to random halves of the pre cohort, where the truth is
"no difference", to check that the rule declares equivalence when it should.

Usage (from repo root):
    python -m analysis.build_rdms.run_k92_equivalence [--n-boot 1000] [--validate]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.build_rdms import kriegeskorte92 as k92
from analysis.build_rdms import model_evaluation as me
from analysis.build_rdms.behavior.spam_ratings import _combine
from analysis.build_rdms.common import load_rdm, subset_condensed
from analysis.build_rdms.run_k92 import RESULTS_DIR, _mean_rdm
from analysis.utils.equivalence import equivalence_test_bootstrap_effect

_STATS = {"kendall_a": "Kendall tau-a", "spearman": "Spearman rho"}


def equivalence_table(
    observed: dict[str, pd.DataFrame], boots: pd.DataFrame, n_a: int, n_b: int,
) -> pd.DataFrame:
    """One TOST row per (pair set, target, statistic). `observed[pairs]` is indexed by target with
    the full-data `<stat>_a` / `<stat>_b` values."""
    rows = []
    for (pairs, model), draws in boots.groupby(["pairs", "model"]):
        for key, label in _STATS.items():
            effect = observed[pairs].at[model, f"{key}_a"] - observed[pairs].at[model, f"{key}_b"]
            result = equivalence_test_bootstrap_effect(
                effect, draws[f"{key}_diff"], n_a=n_a, n_b=n_b, label=f"pre - post, {label}, {model}",
            )
            rows.append({"pairs": pairs, "target": model, "statistic": key, **result})
    return pd.DataFrame(rows)


def _observed(rdm_a: np.ndarray, rdm_b: np.ndarray, means: dict[str, np.ndarray], mask: np.ndarray | None) -> pd.DataFrame:
    def scored(rdm: np.ndarray) -> np.ndarray:
        return rdm if mask is None else np.where(mask, rdm, np.nan)

    rows = {}
    for name, target in means.items():
        a, b = me._stats(scored(rdm_a), target), me._stats(scored(rdm_b), target)
        rows[name] = {f"{k}_a": a[k] for k in me._STAT_KEYS} | {f"{k}_b": b[k] for k in me._STAT_KEYS}
    return pd.DataFrame.from_dict(rows, orient="index")


def run(n_boot: int, seed: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overlap = k92.approved_overlap(k92.build_overlap_manifest(k92.load_stimuli()))
    rows = overlap["manifest_row"].to_numpy()
    means = {name: _mean_rdm(rdms) for name, rdms in k92.external_targets(overlap).items()}

    rdm_pre = subset_condensed(load_rdm("behav_pre_simple"), rows)
    rdm_post = subset_condensed(load_rdm("behav_post_simple"), rows)
    joint = ~np.isnan(rdm_pre) & ~np.isnan(rdm_post)
    print(f"[k92-eq] pairs: {len(rdm_pre)} total, pre {int((~np.isnan(rdm_pre)).sum())}, "
          f"post {int((~np.isnan(rdm_post)).sum())}, both {int(joint.sum())}", flush=True)

    trials_pre, n_conds = me.load_trials_by_participant("pre_shine")
    trials_post, _ = me.load_trials_by_participant("post_shine")
    boots = me.bootstrap_cohort_difference(
        trials_pre, trials_post, means, n_conds, n_boot=n_boot, seed=seed, subset=rows,
        pair_masks={"own": None, "joint": joint},
    )
    observed = {"own": _observed(rdm_pre, rdm_post, means, None), "joint": _observed(rdm_pre, rdm_post, means, joint)}
    table = equivalence_table(observed, boots, len(trials_pre), len(trials_post))
    table.to_csv(RESULTS_DIR / "k92_equivalence.csv", index=False)
    boots.to_csv(RESULTS_DIR / "k92_equivalence_boot.csv", index=False)
    print(table[["pairs", "target", "statistic", "effect", "ci_low", "ci_high", "bound", "min_bound", "equivalent"]]
          .to_string(index=False))


def validate(n_boot: int, n_splits: int, seed: int) -> None:
    """Same procedure on random halves of the pre cohort (a true null)."""
    overlap = k92.approved_overlap(k92.build_overlap_manifest(k92.load_stimuli()))
    rows = overlap["manifest_row"].to_numpy()
    means = {name: _mean_rdm(rdms) for name, rdms in k92.external_targets(overlap).items()}
    trials, n_conds = me.load_trials_by_participant("pre_shine")
    ids = list(trials)
    rng = np.random.default_rng(seed)
    out = []
    for split in range(n_splits):
        order = rng.permutation(len(ids))
        half_a = {ids[i]: trials[ids[i]] for i in order[: len(ids) // 2]}
        half_b = {ids[i]: trials[ids[i]] for i in order[len(ids) // 2:]}
        rdm_a = subset_condensed(_combine([t for v in half_a.values() for t in v], n_conds, "simple")[0], rows)
        rdm_b = subset_condensed(_combine([t for v in half_b.values() for t in v], n_conds, "simple")[0], rows)
        boots = me.bootstrap_cohort_difference(
            half_a, half_b, means, n_conds, n_boot=n_boot, seed=seed + split + 1, subset=rows,
        )
        table = equivalence_table({"own": _observed(rdm_a, rdm_b, means, None)}, boots, len(half_a), len(half_b))
        out.append(table.assign(split=split))
        print(f"[k92-eq] validation split {split} done", flush=True)
    result = pd.concat(out, ignore_index=True)
    result.to_csv(RESULTS_DIR / "k92_equivalence_validation.csv", index=False)
    print(result[["split", "target", "statistic", "effect", "ci_low", "ci_high", "bound", "equivalent"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true", help="run the pre-vs-pre split-half calibration instead")
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()
    if args.validate:
        validate(args.n_boot, args.n_splits, args.seed)
        return
    run(args.n_boot, args.seed)


if __name__ == "__main__":
    main()
