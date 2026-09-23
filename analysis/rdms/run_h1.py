"""
Run Q1 analysis 1 (perceptual RDM vs. the Kiani-Mur tree) for both SHINE cohorts and write tidy
CSVs to analysis/results/hierarchy_comparison/ for analysis/rdms/h1_inference.ipynb to plot.

  1a  masked label-shuffle test, participant bootstrap of the combined RDM, per-participant model
      fit vs. lower noise bound            (analysis.rdms.model_evaluation)
  1b  Baker's gamma vs. the KM tree per (MDS dim, linkage) with shuffle null, and per-level
      HDBSCAN agreement                    (analysis.rdms.hierarchical_clustering)

Family-wise correction: Benjamini-Hochberg within each cohort over the confirmatory p-values
(permutation test on tau-a, plus the gamma shuffle test per linkage at the reference dimension).

Usage (from repo root):
    python -m analysis.rdms.run_h1 [--n-perm 1000] [--n-boot 500] [--ref-dim 10]
"""
from __future__ import annotations

import argparse

import pandas as pd
from statsmodels.stats.multitest import multipletests

from analysis.behavioral.cohort_comparison import MDS_CACHE_DIR
from analysis.rdms import hierarchical_clustering as hcl
from analysis.rdms import model_evaluation as me
from analysis.rdms.common import REPO_ROOT, load_rdm
from analysis.utils.stats import wilcoxon_effect_size

RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "hierarchy_comparison"
_COHORTS = {"pre": "pre_shine", "post": "post_shine"}
_MDS_DIMS = (2, 3, 5, 10)
_KM_LEVELS = (1, 2, 3)


def run_cohort(cohort: str, *, n_perm: int, n_boot: int, ref_dim: int, seed: int) -> dict[str, pd.DataFrame]:
    variant = _COHORTS[cohort]
    behav = load_rdm(f"behav_{cohort}_simple")
    km = load_rdm("sem_km")
    tbp, n_conds = me.load_trials_by_participant(variant)
    out: dict[str, pd.DataFrame] = {}

    shuffle = me.label_shuffle_test_masked(behav, km, n_perm=n_perm, seed=seed)
    out["permutation"] = shuffle.summary().reset_index().assign(cohort=cohort)

    loo = me.loo_model_fit(tbp, km, n_conds)
    out["loo_fit"] = loo.assign(cohort=cohort)

    boot = me.bootstrap_combined_fit(tbp, km, n_conds, n_boot=n_boot, seed=seed)
    out["bootstrap"] = boot.assign(cohort=cohort)

    paired = loo.dropna(subset=["model_kendall_a", "lower_kendall_a"])
    fit_ci = me.bootstrap_mean_ci(paired["model_kendall_a"].to_numpy(), seed=seed)
    lower_ci = me.bootstrap_mean_ci(paired["lower_kendall_a"].to_numpy(), seed=seed)
    combined_ci = me.percentile_ci(boot["kendall_a"].to_numpy())
    wilcoxon = wilcoxon_effect_size(paired["model_kendall_a"].to_numpy(), paired["lower_kendall_a"].to_numpy())
    out["summary"] = pd.DataFrame([{
        "cohort": cohort, "n_participants": len(loo), "n_boot": n_boot, "n_perm": n_perm,
        "combined_tau_a": shuffle.observed["kendall_a"], "combined_tau_a_ci_lo": combined_ci[0],
        "combined_tau_a_ci_hi": combined_ci[1], "combined_spearman": shuffle.observed["spearman"],
        "perm_p_tau_a": shuffle.p_value["kendall_a"],
        "mean_model_tau_a": paired["model_kendall_a"].mean(), "model_ci_lo": fit_ci[0], "model_ci_hi": fit_ci[1],
        "mean_lower_tau_a": paired["lower_kendall_a"].mean(), "lower_ci_lo": lower_ci[0], "lower_ci_hi": lower_ci[1],
        "model_vs_lower_wilcoxon_p": wilcoxon["p_value"], "model_vs_lower_effect": wilcoxon["effect_size"],
    }])

    parts = hcl.path_parts()
    z_gt = hcl.path_tree_to_linkage(parts)
    embedded = hcl.perceptual_embeddings(behav, cohort, _MDS_DIMS, MDS_CACHE_DIR)
    out["agglomerative"] = hcl.agglomerative_agreement(z_gt, embedded).assign(cohort=cohort)

    gamma_rows = []
    for method in out["agglomerative"]["linkage"].unique():
        z = hcl.build_linkage(embedded[ref_dim], method)
        result = hcl.dendrogram_shuffle_test(hcl.cophenetic_ranks(z), parts, n_perm=n_perm, seed=seed)
        gamma_rows.append({
            "cohort": cohort, "ndim": ref_dim, "linkage": method, "baker_gamma": result.observed,
            "null_mean": float(result.null.mean()), "null_sd": float(result.null.std()),
            "p_value": result.p_value,
        })
    out["gamma_shuffle"] = pd.DataFrame(gamma_rows)

    out["hdbscan"] = hcl.hdbscan_level_agreement(embedded, parts, _KM_LEVELS).assign(cohort=cohort)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--ref-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    collected: dict[str, list[pd.DataFrame]] = {}
    for cohort in _COHORTS:
        print(f"[h1] {cohort}: running", flush=True)
        for name, frame in run_cohort(
            cohort, n_perm=args.n_perm, n_boot=args.n_boot, ref_dim=args.ref_dim, seed=args.seed,
        ).items():
            collected.setdefault(name, []).append(frame)

    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in collected.items()}
    tables["fdr"] = family_fdr(tables["permutation"], tables["gamma_shuffle"])
    for name, table in tables.items():
        table.to_csv(RESULTS_DIR / f"h1_{name}.csv", index=False)
        print(f"[h1] wrote h1_{name}.csv ({len(table)} rows)")


def family_fdr(permutation: pd.DataFrame, gamma_shuffle: pd.DataFrame) -> pd.DataFrame:
    """BH-FDR within each cohort over its confirmatory family: the tau-a permutation p plus each
    linkage's Baker's-gamma shuffle p at the reference dimension."""
    rows = []
    for cohort in permutation["cohort"].unique():
        tau = permutation[(permutation["cohort"] == cohort) & (permutation["statistic"] == "kendall_a")]
        family = [{"cohort": cohort, "test": "tau_a_permutation", "p_value": float(tau["p_value"].iloc[0])}]
        for _, row in gamma_shuffle[gamma_shuffle["cohort"] == cohort].iterrows():
            family.append({"cohort": cohort, "test": f"baker_gamma_{row['linkage']}", "p_value": row["p_value"]})
        frame = pd.DataFrame(family)
        frame["p_fdr"] = multipletests(frame["p_value"], method="fdr_bh")[1]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    main()
