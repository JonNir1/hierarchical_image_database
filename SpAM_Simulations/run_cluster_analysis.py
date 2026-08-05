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

Usage (from the repo root)::

    python SpAM_Simulations/run_cluster_analysis.py \\
        --store SpAM_Simulations/sim_results/design-comparison/mds_store \\
        --out   SpAM_Simulations/sim_results/design-comparison/out
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from SpAM_Simulations.cluster_stability import (
    DEFAULT_KS, DEFAULT_LINKAGES, compute_cluster_agreement, compute_cluster_sizes,
    compute_dendrogram_agreement, continuum_diagnostics, select_k,
)
from SpAM_Simulations.storage import ResultStore

# Grouping for k-selection and the continuum verdicts. `allocation_mode` is included when present so
# the two arms get their own k* rather than one averaged over both - the whole point of the sweep is
# that the arms might differ, and pooling them would hide exactly that.
SELECT_BY = ("num_subjects", "allocation_mode", "ndim", "linkage")

# Reported at BOTH chosen granularities. Carrying all three at each k* is what makes the trade
# readable in either direction: how much separation the parsimonious VI choice gives up, and how
# much reproducibility the silhouette choice costs. With only one side you can see that the two k*
# differ but not whether the difference matters.
AT_K_METRICS = ("vi_norm", "sil_cross", "sil_ratio")


def _select_by(agreement: pd.DataFrame) -> list:
    return [c for c in SELECT_BY if c in agreement.columns]


def _metrics_at_k(frame: pd.DataFrame, agreement: pd.DataFrame, by: list,
                  suffix: str) -> pd.DataFrame:
    """Attach each group's ``AT_K_METRICS`` read off the agreement curve at its ``k_star_<suffix>``.

    A left merge on the group key plus k, so a group whose chosen k somehow has no agreement row
    gets NaN rather than silently dropping out of the table.
    """
    k_col = f"k_star_{suffix}"
    cols = [f"mean_{m}" for m in AT_K_METRICS if f"mean_{m}" in agreement.columns]
    lookup = agreement[by + ["k"] + cols].rename(columns={"k": k_col})
    merged = frame.merge(lookup, on=by + [k_col], how="left")
    return merged.rename(columns={f"mean_{m}": f"{m}_at_{k_col}" for m in AT_K_METRICS})


def run(store_path: Path, out_dir: Path, ks: Sequence[int] = DEFAULT_KS,
        linkages: Sequence[str] = DEFAULT_LINKAGES, verbose: bool = True) -> dict:
    """Compute every cluster table and write it under ``out_dir``. Returns the frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore.open(store_path)
    print(f"[store] {store_path}: {len(store)} records, "
          f"confdists {'present' if store.has_confdists else 'ABSENT (recomputing from confs)'}")

    agreement = compute_cluster_agreement(store, ks=ks, linkages=linkages, verbose=verbose)
    agreement.to_csv(out_dir / "cluster_agreement.csv", index=False)

    dendro = compute_dendrogram_agreement(store, linkages=linkages, verbose=verbose)
    dendro.to_csv(out_dir / "dendrogram_agreement.csv", index=False)

    sizes = compute_cluster_sizes(store, ks=ks, linkages=linkages, verbose=verbose)
    sizes.to_csv(out_dir / "cluster_sizes.csv", index=False)

    by = _select_by(agreement)
    # TWO granularities, because VI alone does not identify the number of clusters. VI measures
    # *reproducibility*, and a coarse cut of a well-separated structure is perfectly reproducible
    # too: on three planted blobs VI is exactly 0 at k=2 and at k=3 alike, so the parsimony tiebreak
    # returns 2. Silhouette is what distinguishes them - it peaks at the true 3 (0.93 against 0.76).
    # `k_star_vi` is the safe deduplication rule; `k_star_sil` is the scientific claim.
    #
    # `select_k` and `continuum_diagnostics` take a `criterion=` and emit a generic `k_star`, so the
    # criterion is named here, where the two coexist and a bare `k_star` would be ambiguous.
    diagnostics = continuum_diagnostics(agreement, criterion="vi_norm", by=by)
    # One file, not two: k* is meaningless without the verdicts that say whether it means anything.
    k_selection = diagnostics[by + ["k_star", "vi_norm_range", "is_flat", "is_arbitrary_slicing"]]
    k_selection = k_selection.rename(columns={"k_star": "k_star_vi"}).assign(rule="one_se")

    chosen_sil = select_k(agreement, criterion="sil_cross", by=by)[by + ["k_star"]]
    k_selection = k_selection.merge(chosen_sil.rename(columns={"k_star": "k_star_sil"}),
                                    on=by, how="outer")
    for suffix in ("vi", "sil"):
        k_selection = _metrics_at_k(k_selection, agreement, by, suffix)
    k_selection.to_csv(out_dir / "k_selection.csv", index=False)

    _report(k_selection, dendro)
    return {"cluster_agreement": agreement, "dendrogram_agreement": dendro,
            "cluster_sizes": sizes, "k_selection": k_selection}


def _report(k_selection: pd.DataFrame, dendro: pd.DataFrame) -> None:
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
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    run(args.store, args.out,
        ks=[int(k) for k in args.ks.split(",")],
        linkages=tuple(args.linkages.split(",")),
        verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
