"""Local driver for the cluster-reproducibility analysis (pipeline steps g-i).

Runs **after** an EC2 sweep, on a downloaded ``ResultStore``. Steps a-f (generate cohorts, fit MDS,
store) happen on EC2; everything here is pure post-processing over the stored configurations, needs
no R, and takes a few minutes on one machine.

**Why the store can be downloaded without ``confdists.f32``.** Every distance used here is
recomputed as ``pdist(store.conf(row, ndim))``, which is bit-identical to the stored ``confdist``
row. A conf row is ``n_images * max_ndim`` floats against ``n_images^2 / 2`` for a distance row -
roughly a twentieth of the size - so a 480-fit sweep downloads as ~28 MB rather than ~500 MB.
Recomputing also makes **Ward exactly valid**: Ward's objective is defined only for Euclidean input,
which coordinates are and an arbitrary stored distance vector need not be.

**What this answers.** Not "do two cohorts agree on every distance" (global Spearman, dominated by
the ~262k unrelated pairs) and not "do they agree on which single pair is closest" (top-k Jaccard,
which counts a reshuffle among three near-identical red flowers as three errors when the downstream
decision - pick one of them - is unchanged). It answers: **do two independent cohorts of N subjects
discover the same groups?** That is the question stimulus construction actually poses.

**The null result is a result.** If the space is a continuum rather than lumpy, VI will be high and
flat at every k and no stable k* exists; or the cohorts will agree on a cut that has no separation
in either geometry. ``continuum_diagnostics`` reports both as booleans and this driver prints them
prominently. Either one means "one image per cluster" is the wrong rule and a distance threshold
should be used instead - a finding, not a failure.

**Two clusterers, answering two different questions.** ``cluster_stability`` runs agglomerative
clustering, which assigns *every* image to a cluster; that hard partition is what Variation of
Information needs to be a metric, and metricity is what the later path-hierarchy comparison chains
through. But it also means a genuinely isolated image is absorbed into whichever group is nearest.
``density_clustering`` runs HDBSCAN alongside it purely to recover that missing statement - which
images belong to no group at all - and its outputs are descriptive: they never enter the VI chain.
See the README's "Clustering algorithms" section for why GMM is not among the options.

Usage (from the repo root)::

    python SpAM_Simulations/run_cluster_analysis.py \\
        --store SpAM_Simulations/sim_results/design-comparison/mds_store \\
        --out   SpAM_Simulations/sim_results/design-comparison/out
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from SpAM_Simulations.measures.cluster_stability import (
    DEFAULT_KS, DEFAULT_LINKAGES, DEFAULT_MAX_PAIRS, HIGH_K_THRESHOLD,
    compute_agglomerative_tables, continuum_diagnostics, select_k,
)
from SpAM_Simulations.measures.density_clustering import (
    DEFAULT_MIN_CLUSTER_SIZES, compute_density_agreement, isolated_images,
)
from SpAM_Simulations.core.storage import ResultStore

# Grouping for k-selection and the continuum verdicts. `allocation_mode` is included when present so
# the two arms get their own k* rather than one averaged over both - the whole point of the sweep is
# that the arms might differ, and pooling them would hide exactly that.
# k* is chosen per FULL CONFIGURATION, not per (N, arm, ndim, linkage).
#
# Those four were the intended reporting axes, but the agreement frame carries a row per swept
# parameter combination - softness x min_reliability x dispersion, 36 of them - so grouping by only
# four of the keys leaves 36 rows per (group, k). That silently broke both consumers: `select_k` fed
# `apply_selection_rule` a k-axis with every k repeated 36 times, and `_metrics_at_k` merged against
# a non-unique lookup and multiplied the table by 36 per call. Run twice, that turned 144 groups
# into 186,624 rows and a 35 MB CSV of nonsense.
#
# Averaging the sensitivity axes away first would fix the shape but requires inventing a pooling
# rule for the SEM. Selecting per configuration needs no such invention, gives exactly one row per
# k within each group, and answers the more useful question directly: how stable is k* ACROSS the
# swept range? The reporting axes below are what the distribution of k* is then summarised over.
REPORT_BY = ("num_subjects", "allocation_mode", "ndim", "linkage")
# Columns that are outcomes rather than group keys, so everything else identifies a configuration.
_NON_KEY_COLUMNS = ("k", "n_reps", "n_pairs", "high_k")

# Reported at BOTH chosen granularities. Carrying all three at each k* is what makes the trade
# readable in either direction: how much separation the parsimonious VI choice gives up, and how
# much reproducibility the silhouette choice costs. With only one side you can see that the two k*
# differ but not whether the difference matters.
AT_K_METRICS = ("vi_norm", "sil_cross", "sil_ratio")


def _select_by(agreement: pd.DataFrame) -> list:
    """Every column that identifies a configuration: one row per k within each resulting group."""
    return [c for c in agreement.columns
            if c not in _NON_KEY_COLUMNS and not c.startswith(("mean_", "sem_"))]


def _metrics_at_k(frame: pd.DataFrame, agreement: pd.DataFrame, by: list,
                  suffix: str) -> pd.DataFrame:
    """Attach each group's ``AT_K_METRICS`` read off the agreement curve at its ``k_star_<suffix>``.

    A left merge on the group key plus k, so a group whose chosen k somehow has no agreement row
    gets NaN rather than silently dropping out of the table.
    """
    k_col = f"k_star_{suffix}"
    cols = [f"mean_{m}" for m in AT_K_METRICS if f"mean_{m}" in agreement.columns]
    lookup = agreement[by + ["k"] + cols].rename(columns={"k": k_col})
    # A non-unique lookup turns this left merge into a cartesian product, which is how a 144-row
    # table once became 186,624 rows. Fail loudly instead: it means `by` does not identify a
    # configuration, and every number downstream would be a duplicate.
    duplicated = int(lookup.duplicated(subset=by + [k_col]).sum())
    if duplicated:
        raise ValueError(
            f"the agreement lookup has {duplicated} duplicate rows per {by + [k_col]}, so merging "
            f"would multiply the table. `by` must identify one row per k - check that every swept "
            f"parameter column is included."
        )
    merged = frame.merge(lookup, on=by + [k_col], how="left", validate="one_to_one")
    return merged.rename(columns={f"mean_{m}": f"{m}_at_{k_col}" for m in AT_K_METRICS})


def build_k_selection(agreement: pd.DataFrame) -> pd.DataFrame:
    """Choose a granularity per configuration, with the verdicts that say whether it means anything.

    Derived entirely from the agreement frame, so it can be rebuilt from ``cluster_agreement.csv``
    without touching the store - which matters, because that frame costs hours and this does not.

    TWO granularities, because VI alone does not identify the number of clusters. VI measures
    *reproducibility*, and a coarse cut of a well-separated structure is perfectly reproducible too:
    on three planted blobs VI is exactly 0 at k=2 and at k=3 alike, so the parsimony tiebreak returns
    2. Silhouette is what distinguishes them - it peaks at the true 3 (0.93 against 0.76).
    ``k_star_vi`` is the safe deduplication rule; ``k_star_sil`` is the scientific claim.

    ``select_k`` and ``continuum_diagnostics`` take a ``criterion=`` and emit a generic ``k_star``,
    so the criterion is named here, where the two coexist and a bare ``k_star`` would be ambiguous.
    """
    by = _select_by(agreement)
    diagnostics = continuum_diagnostics(agreement, criterion="vi_norm", by=by)
    # One file, not two: k* is meaningless without the verdicts that say whether it means anything.
    k_selection = diagnostics[by + ["k_star", "vi_norm_range", "is_flat", "is_arbitrary_slicing"]]
    k_selection = k_selection.rename(columns={"k_star": "k_star_vi"}).assign(rule="one_se")

    chosen_sil = select_k(agreement, criterion="sil_cross", by=by)[by + ["k_star"]]
    k_selection = k_selection.merge(chosen_sil.rename(columns={"k_star": "k_star_sil"}),
                                    on=by, how="outer", validate="one_to_one")
    for suffix in ("vi", "sil"):
        k_selection = _metrics_at_k(k_selection, agreement, by, suffix)
    for suffix in ("vi", "sil"):
        col = f"k_star_{suffix}"
        if col in k_selection:
            k_selection[f"{col}_is_high_k"] = k_selection[col] >= HIGH_K_THRESHOLD
    return k_selection


def run(store_path: Path, out_dir: Path, ks: Sequence[int] = DEFAULT_KS,
        linkages: Sequence[str] = DEFAULT_LINKAGES,
        min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES,
        density_mcs: int = 5, verbose: bool = True, n_jobs: int = -1,
        max_pairs: Optional[int] = DEFAULT_MAX_PAIRS) -> dict:
    """Compute every cluster table and write it under ``out_dir``. Returns the frames.

    ``n_jobs`` parallelises over configuration groups, which are independent. ``max_pairs`` caps the
    rep pairs compared per group: at reps=10 the full set is 45 and the pair loop dominates, so the
    default of 22 roughly halves it for a ~1.4x wider SEM on an estimate that was never based on
    independent pairs anyway. Pass ``max_pairs=None`` to exhaust them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore.open(store_path)
    print(f"[store] {store_path}: {len(store)} records, "
          f"confdists {'present' if store.has_confdists else 'ABSENT (recomputing from confs)'}")

    # ONE traversal for all three agglomerative tables. Computed separately, each rebuilt the same
    # linkage trees, cuts and cophenetic rankings for every cohort - three times the dominant cost.
    agglomerative = compute_agglomerative_tables(store, ks=ks, linkages=linkages, verbose=verbose,
                                                 n_jobs=n_jobs, max_pairs=max_pairs)
    agreement = agglomerative["agreement"]
    # The caveat travels with the data rather than living only in a comment: k >= 150 cuts at
    # roughly leaf granularity, where the pilot supports very little structure (gt_diagnostics puts
    # the raw within-`same_leaf` half-split reliability near 0.05), so those rows report on the
    # embedding's smoothness as much as on anything recoverable.
    agreement["high_k"] = agreement["k"] >= HIGH_K_THRESHOLD
    agreement.to_csv(out_dir / "cluster_agreement.csv", index=False)

    dendro = agglomerative["dendrogram"]
    dendro.to_csv(out_dir / "dendrogram_agreement.csv", index=False)

    sizes = agglomerative["sizes"]
    sizes["high_k"] = sizes["k"] >= HIGH_K_THRESHOLD
    sizes.to_csv(out_dir / "cluster_sizes.csv", index=False)

    k_selection = build_k_selection(agreement)
    by = _select_by(agreement)
    k_selection.to_csv(out_dir / "k_selection.csv", index=False)
    for suffix in ("vi", "sil"):
        flag = f"k_star_{suffix}_is_high_k"
        if flag in k_selection and k_selection[flag].any():
            n = int(k_selection[flag].sum())
            print(f"[k] NOTE: k_star_{suffix} lands at >= {HIGH_K_THRESHOLD} in {n} of "
                  f"{len(k_selection)} groups. At 725 images that is <5 images per cluster, which "
                  f"is the granularity the pilot supports least - read those as a statement about "
                  f"the ground truth rather than as a recommended k.")

    # The density pass, which answers what agglomerative clustering structurally cannot: whether an
    # image belongs to no group at all. Descriptive only, and never substituted into the VI chain.
    density = compute_density_agreement(store, min_cluster_sizes=min_cluster_sizes,
                                        verbose=verbose, n_jobs=n_jobs)
    density.to_csv(out_dir / "density_agreement.csv", index=False)
    isolated = isolated_images(store, min_cluster_size=density_mcs, verbose=verbose,
                               n_jobs=n_jobs)
    isolated.to_csv(out_dir / "isolated_images.csv", index=False)

    _report(k_selection, dendro, density, isolated, density_mcs)
    return {"cluster_agreement": agreement, "dendrogram_agreement": dendro,
            "cluster_sizes": sizes, "k_selection": k_selection,
            "density_agreement": density, "isolated_images": isolated}


def _report(k_selection: pd.DataFrame, dendro: pd.DataFrame,
            density: pd.DataFrame = None, isolated: pd.DataFrame = None,
            density_mcs: int = 5) -> None:
    """Print the verdicts, loudly. A continuum must not be reported as a k*."""
    n = len(k_selection)
    flat = int(k_selection["is_flat"].sum())
    arbitrary = int(k_selection["is_arbitrary_slicing"].sum())
    print("\n" + "=" * 78)
    print(f"CONTINUUM DIAGNOSTICS over {n} groups")
    print("=" * 78)
    print(f"  is_flat              {flat:>4} / {n}   VI does not vary across the k grid, so no "
          f"granularity is distinguishably better")
    print(f"  is_arbitrary_slicing {arbitrary:>4} / {n}   cross-cohort silhouette at k* is near "
          f"zero: agreement without separation")
    if flat or arbitrary:
        print("\n  >> At least one group looks like a CONTINUUM rather than a set of clusters.")
        print("  >> That is a finding: 'one image per cluster' would be the wrong deduplication")
        print("  >> rule there, and a distance threshold should be used instead.")
    clean = k_selection[~k_selection["is_flat"] & ~k_selection["is_arbitrary_slicing"]]
    if len(clean):
        print(f"\n  {len(clean)} group(s) with a meaningful k*:")
        cols = [c for c in ("num_subjects", "allocation_mode", "ndim", "linkage",
                            "k_star_vi", "vi_norm_at_k_star_vi", "sil_cross_at_k_star_vi",
                            "k_star_sil", "vi_norm_at_k_star_sil", "sil_cross_at_k_star_sil")
                if c in clean.columns]
        print(clean[cols].to_string(index=False))
        print("  k_star_vi  = coarsest granularity that REPRODUCES (VI, one-SE) - the safe "
              "deduplication rule")
        print("  k_star_sil = granularity with the most cross-cohort SEPARATION - where the "
              "structure is")
        print("  Both are scored at BOTH k*, so the trade reads in either direction: what the "
              "parsimonious")
        print("  choice gives up in separation, and what the structured choice costs in "
              "reproducibility.")
    if len(dendro):
        gcol = "mean_baker_gamma"
        if gcol in dendro.columns:
            print(f"\n  Baker's gamma (k-free, per linkage): "
                  f"{dendro.groupby('linkage')[gcol].mean().round(3).to_dict()}")

    # The density pass. Reported separately and never mixed into the VI numbers above, because a
    # labelling with a noise class is not a partition and its agreement scores do not compose.
    if density is not None and len(density):
        print("\n" + "-" * 78)
        print("DENSITY PASS (HDBSCAN, descriptive - NOT part of the VI/transitivity chain)")
        print("-" * 78)
        cols = [c for c in ("min_cluster_size", "mean_n_clusters", "mean_frac_noise",
                            "mean_noise_kappa", "mean_noise_jaccard",
                            "mean_ari_shared_clustered", "mean_frac_shared_clustered")
                if c in density.columns]
        print(density.groupby("min_cluster_size", as_index=False)[
            [c for c in cols if c != "min_cluster_size"]].mean().round(3).to_string(index=False))
        print("  frac_noise  = images HDBSCAN found confusable with nothing - the ones agglomerative")
        print("                clustering is forced to absorb into some cluster regardless")
        print("  noise_kappa = do two cohorts agree on WHICH images those are (chance-corrected)")
    if isolated is not None and len(isolated):
        always = float((isolated["frac_cohorts_noise"] == 1.0).mean())
        never = float((isolated["frac_cohorts_noise"] == 0.0).mean())
        print(f"\n  Per-image isolation at min_cluster_size={density_mcs}: "
              f"{always:.1%} of images unclustered in EVERY cohort, {never:.1%} in none.")
        print("  The first group is the safest to use as stimuli; the rest need their group-mates "
              "excluded.")
    print("=" * 78 + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True, type=Path, help="downloaded mds_store/ directory")
    p.add_argument("--out", required=True, type=Path, help="directory to write the CSVs into")
    p.add_argument("--ks", type=str, default=",".join(str(k) for k in DEFAULT_KS),
                   help="comma-separated cluster counts to cut at")
    p.add_argument("--linkages", type=str, default=",".join(DEFAULT_LINKAGES),
                   help="comma-separated agglomerative linkages")
    p.add_argument("--min-cluster-sizes", type=str,
                   default=",".join(str(m) for m in DEFAULT_MIN_CLUSTER_SIZES),
                   help="comma-separated HDBSCAN min_cluster_size values (density pass)")
    p.add_argument("--density-mcs", type=int, default=5,
                   help="min_cluster_size used for the per-image isolated_images.csv table")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="parallel workers over configuration groups; 1 runs in-process")
    p.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
                   help="rep pairs compared per group (0 or negative = all C(n_reps, 2))")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    run(args.store, args.out,
        ks=[int(k) for k in args.ks.split(",")],
        linkages=tuple(args.linkages.split(",")),
        min_cluster_sizes=[int(m) for m in args.min_cluster_sizes.split(",")],
        density_mcs=args.density_mcs,
        verbose=not args.quiet,
        n_jobs=args.n_jobs,
        # 0 or negative means "no cap", which argparse cannot express as None directly.
        max_pairs=args.max_pairs if args.max_pairs and args.max_pairs > 0 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
