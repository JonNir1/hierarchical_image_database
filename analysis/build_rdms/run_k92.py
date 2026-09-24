"""
Run Q1 analysis 2 (perceptual RDM vs. the external Kriegeskorte 2008b / Mur 2013 data) on the
visually approved overlap images, for both SHINE cohorts, and write tidy CSVs to
analysis/results/kriegeskorte92/.

  targets   -- Mur 2013 similarity judgments (primary), human IT fMRI and monkey IT (secondary),
               each averaged over its participants/sessions.
  RDM level -- masked image-relabeling permutation test (tau-a primary), participant bootstrap of our
               combined RDM, and bootstrap over the external participants' own fit.
  trees     -- Baker's gamma between our agglomerative trees (cached MDS embedding of all 725 images,
               restricted to the overlap) and the external behavioral tree, with shuffle nulls; the
               KM tree restricted to the same images is included for context. HDBSCAN vs KM levels
               is descriptive only at this N.

BH-FDR within each cohort over the confirmatory family (three permutation tests + the two matched-
linkage gamma tests at the reference dimension).

Usage (from repo root):
    python -m analysis.build_rdms.run_k92 [--n-perm 5000] [--n-boot 500] [--ref-dim 10]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from analysis.pre_vs_post_shine.geometry import MDS_CACHE_DIR
from analysis.build_rdms import hierarchical_clustering as hcl
from analysis.build_rdms import kriegeskorte92 as k92
from analysis.build_rdms import model_evaluation as me
from analysis.build_rdms.common import REPO_ROOT, load_rdm, subset_condensed
from analysis.utils.stats import masked_rank_correlation

RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "kriegeskorte92"
_COHORTS = {"pre": "pre_shine", "post": "post_shine"}
_MDS_DIMS = (2, 3, 5, 10)
_PRIMARY = "behavioral_mur2013"
_MATCHED_LINKAGES = ("average", "complete")
_HDBSCAN_SIZES = (2, 3, 5)


def _mean_rdm(rdms: list[np.ndarray]) -> np.ndarray:
    return np.nanmean(np.vstack(rdms), axis=0)


def _external_leave_one_out(rdms: list[np.ndarray]) -> float:
    """Mean tau-a between each external participant and the mean of the others: the ceiling for
    how well any single external participant can be predicted by the group."""
    values = [
        masked_rank_correlation(r, _mean_rdm(rdms[:i] + rdms[i + 1:]), tau_a=True)["kendall_a"]
        for i, r in enumerate(rdms)
    ]
    return float(np.mean(values))


def run_cohort(
    cohort: str, overlap: pd.DataFrame, targets: dict[str, list[np.ndarray]],
    *, n_perm: int, n_boot: int, ref_dim: int, seed: int,
) -> dict[str, pd.DataFrame]:
    rows = overlap["manifest_row"].to_numpy()
    behav_full = load_rdm(f"behav_{cohort}_simple")
    perceptual = subset_condensed(behav_full, rows)
    km = subset_condensed(load_rdm("sem_km"), rows)
    tbp, n_conds = me.load_trials_by_participant(_COHORTS[cohort])
    out: dict[str, pd.DataFrame] = {}

    means = {name: _mean_rdm(rdms) for name, rdms in targets.items()}
    boots = me.bootstrap_combined_fit_multi(tbp, means, n_conds, n_boot=n_boot, seed=seed, subset=rows)
    summary, permutation = [], []
    for name, rdms in targets.items():
        target = means[name]
        shuffle = me.label_shuffle_test_masked(perceptual, target, n_perm=n_perm, seed=seed)
        combined_ci = me.percentile_ci(boots.loc[boots["model"] == name, "kendall_a"].to_numpy())
        per_external = np.array([masked_rank_correlation(perceptual, r, tau_a=True)["kendall_a"] for r in rdms])
        ext_ci = me.bootstrap_mean_ci(per_external, seed=seed) if len(rdms) > 1 else (np.nan, np.nan)
        permutation.append(shuffle.summary().reset_index().assign(cohort=cohort, target=name))
        summary.append({
            "cohort": cohort, "target": name, "n_images": len(rows), "n_pairs": len(perceptual),
            "n_pairs_observed": shuffle.observed["n_overlap"], "n_external_rdms": len(rdms),
            "tau_a": shuffle.observed["kendall_a"], "spearman": shuffle.observed["spearman"],
            "perm_p_tau_a": shuffle.p_value["kendall_a"],
            "boot_ours_ci_lo": combined_ci[0], "boot_ours_ci_hi": combined_ci[1],
            "mean_per_external_tau_a": float(np.nanmean(per_external)),
            "boot_external_ci_lo": ext_ci[0], "boot_external_ci_hi": ext_ci[1],
            "external_loo_tau_a": _external_leave_one_out(rdms) if len(rdms) > 1 else np.nan,
            "external_vs_km_tau_a": masked_rank_correlation(_mean_rdm(rdms), km, tau_a=True)["kendall_a"],
            "ours_vs_km_tau_a": masked_rank_correlation(perceptual, km, tau_a=True)["kendall_a"],
        })
    out["summary"] = pd.DataFrame(summary)
    out["permutation"] = pd.concat(permutation, ignore_index=True)

    parts = [hcl.path_parts()[r] for r in rows]
    z_km = hcl.path_tree_to_linkage(parts)
    ext_mean = _mean_rdm(targets[_PRIMARY])
    ext_trees = {m: hcl.build_linkage(ext_mean, m) for m in _MATCHED_LINKAGES}
    embedded = {
        d: subset_condensed(e, rows)
        for d, e in hcl.perceptual_embeddings(behav_full, cohort, _MDS_DIMS, MDS_CACHE_DIR).items()
    }

    gamma, gamma_shuffle = [], []
    for dim, dists in embedded.items():
        for ours_linkage in hcl.DEFAULT_LINKAGES:
            z_ours = hcl.build_linkage(dists, ours_linkage)
            for ext_linkage, z_ext in ext_trees.items():
                gamma.append({
                    "cohort": cohort, "ndim": dim, "ours_linkage": ours_linkage, "ext_linkage": ext_linkage,
                    "baker_gamma_vs_external": hcl.baker_gamma(hcl.cophenetic_ranks(z_ours), hcl.cophenetic_ranks(z_ext)),
                    "baker_gamma_vs_km": hcl.baker_gamma(hcl.cophenetic_ranks(z_ours), hcl.cophenetic_ranks(z_km)),
                })
    for method in _MATCHED_LINKAGES:
        z_ours = hcl.build_linkage(embedded[ref_dim], method)
        result = hcl.tree_shuffle_test(z_ours, ext_trees[method], n_perm=n_perm, seed=seed)
        gamma_shuffle.append({
            "cohort": cohort, "test": f"ours_vs_external_{method}", "ndim": ref_dim, "baker_gamma": result.observed,
            "null_mean": float(result.null.mean()), "null_sd": float(result.null.std()), "p_value": result.p_value,
        })
    context = hcl.tree_shuffle_test(z_km, ext_trees["average"], n_perm=n_perm, seed=seed)
    gamma_shuffle.append({
        "cohort": cohort, "test": "km_vs_external_average", "ndim": np.nan, "baker_gamma": context.observed,
        "null_mean": float(context.null.mean()), "null_sd": float(context.null.std()), "p_value": context.p_value,
    })
    out["gamma"] = pd.DataFrame(gamma)
    out["gamma_shuffle"] = pd.DataFrame(gamma_shuffle)
    out["hdbscan"] = hcl.hdbscan_level_agreement(
        embedded, parts, (1, 2, 3), _HDBSCAN_SIZES,
    ).assign(cohort=cohort)
    return out


def family_fdr(permutation: pd.DataFrame, gamma_shuffle: pd.DataFrame) -> pd.DataFrame:
    """BH-FDR within each cohort: the three tau-a permutation p-values plus the two matched-linkage
    gamma-vs-external p-values. The KM-vs-external gamma is context, not part of the family."""
    frames = []
    for cohort in permutation["cohort"].unique():
        tau = permutation[(permutation["cohort"] == cohort) & (permutation["statistic"] == "kendall_a")]
        family = [{"cohort": cohort, "test": f"tau_a_permutation_{r.target}", "p_value": r.p_value} for r in tau.itertuples()]
        gamma = gamma_shuffle[(gamma_shuffle["cohort"] == cohort) & gamma_shuffle["test"].str.startswith("ours_vs_external")]
        family += [{"cohort": cohort, "test": r.test, "p_value": r.p_value} for r in gamma.itertuples()]
        frame = pd.DataFrame(family)
        frame["p_fdr"] = multipletests(frame["p_value"], method="fdr_bh")[1]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--ref-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overlap = k92.approved_overlap(k92.build_overlap_manifest(k92.load_stimuli()))
    overlap.to_csv(RESULTS_DIR / "approved_matches.csv", index=False)
    print(f"[k92] {len(overlap)} approved overlap images")
    targets = k92.external_targets(overlap)

    collected: dict[str, list[pd.DataFrame]] = {}
    for cohort in _COHORTS:
        print(f"[k92] {cohort}: running", flush=True)
        for name, frame in run_cohort(
            cohort, overlap, targets, n_perm=args.n_perm, n_boot=args.n_boot, ref_dim=args.ref_dim, seed=args.seed,
        ).items():
            collected.setdefault(name, []).append(frame)

    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in collected.items()}
    tables["fdr"] = family_fdr(tables["permutation"], tables["gamma_shuffle"])
    for name, table in tables.items():
        table.to_csv(RESULTS_DIR / f"k92_{name}.csv", index=False)
        print(f"[k92] wrote k92_{name}.csv ({len(table)} rows)")


if __name__ == "__main__":
    main()
