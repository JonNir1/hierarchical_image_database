"""
Generalizability comparison for the three behavioral-RDM aggregation methods
('mean' / 'simple' / 'evidence', see analysis/rdms/behavioral.py) -- the second,
independent line of evidence alongside analysis/behavioral/reliability.py's
within-cohort noise-ceiling comparison (leave-one-out + split-half).

Where reliability.py asks "how much does resampling *within* the retained
cohort agree with itself", this module asks "does the RDM built from the
retained cohort agree with an RDM built from subjects who were never part of
it": pilot-era task versions (1.0/2.0/3.0/3.06), plus v4+ subjects with
status 'screened out' or 'false_positive'. A method that's reliable *and*
generalizes to this independent, unscreened data is a stronger candidate than
one that's merely reliable within its own sample.

Out-of-sample (OOS) pool definition (see _oos_participant_ids()): NOT in the
retained cohort (status=='full data' & task_version>=4.0, same as
analysis.rdms.behavioral._select_participants()) AND status not in
{'revoked consent', 'missing data'} -- those two are hard exclusions
regardless of purpose (revoked consent is an ethical/legal boundary, not a
data-quality one; missing data has no content to use). Confirmed against the
real cached data: pre-SHINE OOS N=63 (42 pilot + 11 screened-out + 10
false-positive), 43.22% pair coverage; post-SHINE OOS N=18 (6 pilot + 7
screened-out + 5 false-positive), 15.65% pair coverage. Both are single
connected components -- post-SHINE's much sparser coverage means its MDS leg
(below) should be treated as lower-confidence, not suppressed.

Known provenance note on 'false_positive': that status comes from
_fails_full_session_recheck()/_load_trials_and_flag_false_positive() in
analysis/utils/parser.py (added in commit 39e3b70, merged into main via
claude/friendly-torvalds-0682eb) -- a full-session (not just screening-block)
re-audit that can downgrade a 'full data' participant after the fact. This
module queries whatever 'status' values are actually present in the loaded
cache directly, rather than re-deriving that classification.

Two comparisons, both per (cohort, aggregation):

  raw RDM generalizability -- Kendall's tau + Spearman's rho between the
      analytical RDM (build_behavioral_rdm(), retained cohort) and the OOS
      RDM (build_oos_rdm(), same aggregation method), over jointly-covered
      pairs (reuses analysis.utils.stats.masked_rank_correlation()).
      Unlike reliability.py's LOO/split-half, there is no resampling here --
      exactly one analytical RDM and one OOS RDM per (cohort, aggregation),
      so this is a single point-estimate comparison, not a distribution of
      paired folds (no Wilcoxon test is meaningful here).
  MDS-embedding generalizability -- Procrustes disparity + confdist Pearson r
      between the two RDMs' independently-fitted MDS configurations (R
      backend, via SpAM_Simulations.core.multi_dimensional_scaling.run_mds).
      Unlike reliability.py's noise-ceiling analysis (where a single held-out
      participant's own trial graph is provably disconnected into ~18
      islands, so MDS can never run there), both OOS pools here are pooled,
      multi-subject data and confirmed single connected components, so MDS is
      feasible. Compute cost is also trivial (a handful of one-time fits, not
      the S=100-iteration budget that ruled out MDS for reliability.py's
      split-half leg), so this sweeps ndim in {2, 3, 5, 10}, matching the
      earlier R-vs-rsatoolbox deep-dive's grid (analysis/rsatoolbox_eval/).

The R-dependent functions (compare_generalizability_mds(),
run_generalizability_mds_comparison()) do a *lazy* import of run_mds inside
the function body (mirroring SpAM_Simulations/empirical/gt_construction.py's
own lazy-import pattern at gt_construction.py:214), so importing this module
never touches R/rpy2 -- only calling those two functions does. Callers that
want to guard against R not being configured should probe before calling
them (see analysis/behavioral/aggregation_comparison.ipynb's Part 2 for the
exact pattern).

As with reliability.py, this module is data-generation only -- no stats
significance-testing or plotting here; that's notebook work.

Usage (from repo root, with the main checkout's .venv active):
    python -m analysis.behavioral.generalizability --cache-dir <path to analysis/results/parsed_data>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.behavioral.reliability import (
    _DEFAULT_CACHE_DIR, _DEFAULT_MIN_OVERLAP, RESULTS_DIR, _VALID_VARIANTS, _cohort_name,
)
from analysis.rdms.behavioral import (
    _AGGREGATIONS, _build_trials, _combine, _select_participants, build_behavioral_rdm,
)
from analysis.rdms.common import image_manifest_index, save_rdm
from analysis.rdms.mds import compare_mds_configs
from analysis.utils.load_data import load_data
from analysis.utils.stats import masked_rank_correlation
from SpAM_Simulations.core.helpers import condensed_rdm_to_weighted_dists

# Plain env-var default -- never touches R/rpy2 itself, safe to run unconditionally at
# import time (matches analysis/rsatoolbox_eval/mds_rsatoolbox_compare.py:56).
os.environ.setdefault("R_LIBS_USER", r"C:\Users\nirjo\R_library\4.5")

_EXCLUDED_STATUSES = ("revoked consent", "missing data")
_DEFAULT_DIMS = (2, 3, 5, 10)
_DEFAULT_MAX_ITERS = 500
_DEFAULT_CONVERGENCE_TOL = 1e-6


def _oos_participant_ids(df_participants: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Out-of-sample participant rows for one SHINE variant: same shine_variant, not in
    the retained cohort, and not one of the hard-excluded statuses. See module docstring."""
    shine_value = variant.removesuffix("_shine")
    retained_ids = set(_select_participants(df_participants, variant)["participant_id"])
    mask = (
        (df_participants["shine_variant"] == shine_value)
        & (~df_participants["participant_id"].isin(retained_ids))
        & (~df_participants["status"].isin(_EXCLUDED_STATUSES))
    )
    return df_participants[mask]


def build_oos_rdm(
    variant: str,
    aggregation: str,
    *,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    save: bool = True,
) -> tuple[np.ndarray, dict]:
    """Build the out-of-sample RDM for one (variant, aggregation). Returns
    (condensed, info) where info has n_oos_participants/n_oos_trials/pct_oos_coverage/
    oos_status_breakdown. If save, persists via save_rdm(name=f"behav_{cohort}_
    {aggregation}_oos", ...) -- a first-class, reusable artifact like every other RDM in
    this project, not a throwaway intermediate.

    Loads with prod_only=False (unlike everywhere else in this project) since the pilot
    cohort (cohort=='pilot') is only present in the unfiltered load.
    """
    data = load_data(cache_dir, prod_only=False)
    oos = _oos_participant_ids(data["participants"], variant)

    image_to_idx, n_conds = image_manifest_index()

    trial_data = _build_trials(
        data["trials"], oos["participant_id"], variant, image_to_idx, n_conds,
    )
    condensed, n_obs_per_pair = _combine(trial_data, n_conds, aggregation)

    n_uncovered = int(np.sum(n_obs_per_pair == 0))
    pct_covered = 100.0 * (1 - n_uncovered / n_obs_per_pair.size)
    status_breakdown = {str(k): int(v) for k, v in oos["status"].value_counts().items()}
    cohort = _cohort_name(variant)

    print(
        f"[generalizability] OOS {cohort}-SHINE: {len(oos)} participants "
        f"{status_breakdown}, {len(trial_data)} non-catch trials, "
        f"{pct_covered:.2f}% pair coverage."
    )

    info = {
        "cohort": cohort,
        "aggregation": aggregation,
        "n_oos_participants": int(len(oos)),
        "n_oos_trials": int(len(trial_data)),
        "pct_oos_coverage": pct_covered,
        "oos_status_breakdown": status_breakdown,
    }
    if save:
        short = f"behav_{cohort}_{aggregation}_oos"
        metric = "spam_plain_mean_oos" if aggregation == "mean" else f"spam_multi_arrangement_{aggregation}_oos"
        save_rdm(
            short, condensed, metric=metric, source="analysis.behavioral.generalizability",
            extra={"variant": variant, **info},
        )
        print(f"[generalizability] Saved D_{short}.npy")

    return condensed, info


def compare_generalizability(
    variant: str,
    aggregation: str,
    *,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    min_overlap: int = _DEFAULT_MIN_OVERLAP,
    save_oos: bool = True,
) -> dict:
    """Analytical RDM (retained cohort) vs OOS RDM, Kendall + Spearman over jointly-
    covered pairs. Returns the merged OOS-pool info + correlation stats dict."""
    analytical = build_behavioral_rdm(variant, cache_dir=cache_dir, aggregation=aggregation)
    oos_condensed, oos_info = build_oos_rdm(
        variant, aggregation, cache_dir=cache_dir, save=save_oos,
    )
    stats = masked_rank_correlation(analytical, oos_condensed, min_overlap=min_overlap)
    return {**oos_info, **stats}


def run_generalizability_comparison(
    *,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    min_overlap: int = _DEFAULT_MIN_OVERLAP,
    save_oos: bool = True,
) -> pd.DataFrame:
    """Raw-RDM generalizability for both variants x all 3 aggregations. Returns a tidy
    long-format DataFrame: [cohort, aggregation, metric, value, n_overlap, underpowered,
    n_oos_participants, n_oos_trials, pct_oos_coverage, oos_status_breakdown], metric in
    {'kendall', 'spearman'}. Exactly one row pair (kendall+spearman) per (cohort,
    aggregation) -- no resampling, see module docstring.
    """
    rows = [
        compare_generalizability(
            variant, aggregation, cache_dir=cache_dir, min_overlap=min_overlap, save_oos=save_oos,
        )
        for variant in _VALID_VARIANTS
        for aggregation in _AGGREGATIONS
    ]
    wide = pd.DataFrame(rows)
    id_vars = [c for c in wide.columns if c not in ("kendall", "spearman")]
    long = wide.melt(id_vars=id_vars, value_vars=["kendall", "spearman"], var_name="metric", value_name="value")
    n_underpowered = int(wide["underpowered"].sum())
    if n_underpowered:
        print(f"[generalizability] WARNING: {n_underpowered} (cohort, aggregation) comparisons "
              f"had fewer than {min_overlap} jointly-covered pairs and were flagged underpowered.")
    return long


# --------------------------------------------------------------------------- MDS leg (R)

def compare_generalizability_mds(
    variant: str,
    aggregation: str,
    ndim: int,
    *,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    max_iters: int = _DEFAULT_MAX_ITERS,
    convergence_tol: float = _DEFAULT_CONVERGENCE_TOL,
    precalc_init: bool = True,
) -> dict:
    """MDS-embedding generalizability at one ndim: fit the analytical RDM and the OOS RDM
    independently (R backend), Procrustes-compare the two configurations. Requires R/rpy2/
    smacof to be configured -- the import is deferred to this call, not module load."""
    from SpAM_Simulations.core.multi_dimensional_scaling import run_mds  # lazy: imports R

    analytical = build_behavioral_rdm(variant, cache_dir=cache_dir, aggregation=aggregation)
    oos_condensed, _ = build_oos_rdm(variant, aggregation, cache_dir=cache_dir, save=False)

    dists_a, weight_a = condensed_rdm_to_weighted_dists(analytical)
    dists_b, weight_b = condensed_rdm_to_weighted_dists(oos_condensed)

    out_a = run_mds(dists_a, weight_a, ndim, max_iters, convergence_tol, precalc_init, verbose=False)
    out_b = run_mds(dists_b, weight_b, ndim, max_iters, convergence_tol, precalc_init, verbose=False)

    comparison = compare_mds_configs(out_a, out_b, ndim)
    return {
        "cohort": _cohort_name(variant), "aggregation": aggregation, "ndim": ndim,
        "niter_analytical": out_a["niter"], "niter_oos": out_b["niter"],
        "stress_analytical": out_a["stress"], "stress_oos": out_b["stress"],
        "needs_more_iters_analytical": out_a["needs_more_iters"],
        "needs_more_iters_oos": out_b["needs_more_iters"],
        **comparison,
    }


def run_generalizability_mds_comparison(
    *,
    dims: tuple[int, ...] = _DEFAULT_DIMS,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    max_iters: int = _DEFAULT_MAX_ITERS,
    convergence_tol: float = _DEFAULT_CONVERGENCE_TOL,
    precalc_init: bool = True,
) -> pd.DataFrame:
    """MDS-embedding generalizability for both variants x all 3 aggregations x dims.
    Requires R -- see module docstring for the lazy-import contract. Skips (with a printed
    message) any (variant, aggregation, ndim) whose RDM turns out disconnected rather than
    failing the whole sweep, though both OOS pools were confirmed connected on real data."""
    rows = []
    for variant in _VALID_VARIANTS:
        for aggregation in _AGGREGATIONS:
            for ndim in dims:
                try:
                    rows.append(compare_generalizability_mds(
                        variant, aggregation, ndim, cache_dir=cache_dir, max_iters=max_iters,
                        convergence_tol=convergence_tol, precalc_init=precalc_init,
                    ))
                except RuntimeError as e:
                    print(f"[generalizability] SKIP {variant}/{aggregation}/ndim={ndim}: {e}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--min-overlap", type=int, default=_DEFAULT_MIN_OVERLAP)
    parser.add_argument("--mds", action="store_true", help="Also run the R-backed MDS generalizability leg.")
    parser.add_argument("--dims", type=int, nargs="+", default=list(_DEFAULT_DIMS))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = run_generalizability_comparison(cache_dir=args.cache_dir, min_overlap=args.min_overlap)
    raw_path = RESULTS_DIR / "generalizability.csv"
    raw.to_csv(raw_path, index=False)
    print(f"[generalizability] Saved {len(raw)} rows to {raw_path}")

    if args.mds:
        mds = run_generalizability_mds_comparison(dims=tuple(args.dims), cache_dir=args.cache_dir)
        mds_path = RESULTS_DIR / "generalizability_mds.csv"
        mds.to_csv(mds_path, index=False)
        print(f"[generalizability] Saved {len(mds)} rows to {mds_path}")


if __name__ == "__main__":
    main()
