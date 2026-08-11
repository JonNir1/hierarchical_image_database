"""Read-only ingestion of SpAM pilot data and calibration of the task-v3 simulation to it.

The task-v3 simulation has two free internals - ``subjects_noise_scale`` (within-subject canvas
placement noise) and ``perspective_dispersion`` (between-subject disagreement) - plus a ground-
truth geometry. This module anchors all three to the real pilot:

* **Ground truth** is the weighted-MDS embedding of *all* pooled pilot subjects (see
  :func:`pilot_aggregate` + ``multi_dimensional_scaling.run_mds``); its spectrum and cluster
  structure are inherited, so the synthetic ``decay``/``n_clusters`` knobs become moot.
* **Noise** is pinned by within-subject **test-retest** reliability (the v3.0 whole-trial repeats).
  In the canvas-placement-noise model this is *perspective-invariant* - a whole-trial repeat
  re-projects to the same 2-D arrangement and differs only by fresh placement noise, so test-retest
  is governed by ``subjects_noise_scale`` alone (see :func:`within_subject_test_retest`).
* **Perspective** is then pinned by **between-subject agreement** (:func:`between_subject_agreement`)
  with noise held fixed.

Identifiability is therefore sequential and exact (a triangular system): test-retest -> noise, then
agreement -> dispersion.

Session loading and completion filtering are delegated to ``analysis.utils.parser.load_data``,
which reads a **flat** ``data/`` directory and derives each session's ``cohort`` from its own
``deployment_mode`` rather than from which folder it sits in; ``parse_pairwise_distances`` (also from
``analysis.utils.parser``) parses the per-trial JSON. This module only reduces those trials to the
condensed per-pair distances the calibration needs.

**Simulations use the pilot cohort only.** Calibrating on the live study's data would let the
sample-size and screening conclusions be shaped by the very cohort they are meant to plan - circular,
and equivalent to peeking at the running experiment. :func:`load_pilot_subjects` therefore defaults to
``cohorts=("pilot",)``. Because every v4.0 session is ``production``, that view contains no
screening-block data: reliability comes from v3.x whole-trial repeats, which sit anywhere in the
session rather than in a dedicated screening stage.

**Cohort is not a proxy for SHINE variant.** The pilot cohort is *not* pre-SHINE only: of the 47
loadable pilot subjects, 41 are ``pre`` and 6 are ``post`` (all v3.06), contradicting the task's
own documentation. Ground-truth construction must pass ``variants=("pre",)`` explicitly; noise-model
fitting may use both, since it estimates a property of subjects rather than of the stimulus set.

Nothing here writes participant data or participant-derived artifacts; ``data/`` is human-subjects
data and must stay local (never committed). Distances come straight from each trial's ``pairwise_distances``
(already canvas-diagonal-normalised in [0, 1], so comparable across subjects' screen sizes).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.utils.parser import parse_pairwise_distances
from analysis.utils.parser import load_data
from SpAM_Simulations.models.experiment import _condensed_pair_indices
from SpAM_Simulations.empirical.gt_construction import (
    aggregate_subjects, build_gt, n_components as gt_n_components,
)

# A pilot stimulus path is ``./images/<variant>_shine/<relpath>``; the manifest lists ``<relpath>``.
_IMG_PREFIX = re.compile(r"^\./images/[^/]+/")


def load_manifest(manifest_path: str) -> Tuple[List[str], Dict[str, int]]:
    """Return the manifest's ordered object-image list and a ``{relpath: index}`` map (0..N-1).

    The ordering is the canonical image index used everywhere downstream (and matches the index a
    ground-truth embedding's rows must follow).
    """
    with open(manifest_path, encoding="utf-8") as fh:
        images = json.load(fh)["images"]
    return images, {rel: i for i, rel in enumerate(images)}


def _src_to_relpath(src: str) -> str:
    """``./images/pre_shine/animate/.../fox1.png`` -> ``animate/.../fox1.png``."""
    return _IMG_PREFIX.sub("", src)


@dataclass
class PilotSubject:
    """One completed pilot session, reduced to what calibration needs."""
    participant_id: str
    task_version: float
    n_images: int
    distances: np.ndarray          # condensed (N(N-1)/2,), per-pair mean distance, NaN = unobserved
    n_obs: np.ndarray              # condensed, integer observation count per pair
    retest_pairs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    qc_flag_rate: float = 0.0      # fraction of this subject's main trials flagged by the task QC
    shine_variant: str = ""        # 'pre' | 'post' | '' when the session didn't record one

    def num_observed_pairs(self) -> int:
        return int(np.count_nonzero(self.n_obs))


def _pair_condensed_indices(pairwise_json: str, rel2idx: Dict[str, int]) -> Dict[int, float]:
    """One trial's ``pairwise_distances`` -> ``{condensed_pair_index: distance}`` on the manifest space.

    Reuses ``analysis.utils.parser.parse_pairwise_distances`` (which returns ``{(src1, src2): dist}``
    keyed by image path) and maps each path onto the manifest index, then to its condensed position.
    """
    items = parse_pairwise_distances(pairwise_json)
    if not items:
        return {}
    N = len(rel2idx)
    (paths, dists) = zip(*items.items())
    a = np.fromiter((rel2idx[_src_to_relpath(s1)] for s1, _ in paths), dtype=np.int64, count=len(paths))
    b = np.fromiter((rel2idx[_src_to_relpath(s2)] for _, s2 in paths), dtype=np.int64, count=len(paths))
    cond = _condensed_pair_indices(a, b, N)
    return {int(c): float(d) for c, d in zip(cond, dists)}


def subject_from_trials(trials: pd.DataFrame, rel2idx: Dict[str, int]) -> PilotSubject:
    """Build one :class:`PilotSubject` from a single participant's rows of the parser's trials frame.

    ``trials`` is one participant's slice of ``analysis.utils.parser.load_data(...)["trials"]``,
    joined to ``["participants"]`` for ``task_version`` (columns ``pairwise_distances``, ``trial_id``,
    ``repeat_of_trial``, ``is_catch``, ``qc_flag``, ``task_version``, ``participant_id``). Each trial's
    normalised pairwise distances are accumulated into a per-subject condensed sum/count (a verbatim
    repeat adds a second observation of the same pairs, so the stored value is their mean); repeat
    trials are aligned to their originals via ``repeat_of_trial``, which already holds the original's
    ``trial_id`` - replacing the old two-column ``is_trial_repeat`` / ``repeat_of_trial_number`` scheme.

    Catch trials are dropped explicitly. Under the old parser they were harmlessly self-cancelling
    (their openmoji stimuli are absent from the manifest, so every pair resolved to nothing), but that
    was incidental rather than intended, and ``is_catch`` now makes it explicit.

    ``shine_variant`` is carried through when present so callers can separate the two image variants.
    It has to be recorded on the subject because :func:`_src_to_relpath` strips the ``<variant>_shine``
    path segment, mapping both variants onto the same manifest index - which is what makes silently
    pooling them possible. Absent column -> ``""`` (hand-built frames in tests).
    """
    N = len(rel2idx)
    n_pairs = N * (N - 1) // 2
    total = np.zeros(n_pairs, dtype=np.float64)
    count = np.zeros(n_pairs, dtype=np.int32)

    main = trials[~trials["is_catch"].astype(bool)]
    by_trial_id: Dict[int, Dict[int, float]] = {}
    for _, row in main.iterrows():
        pairs = _pair_condensed_indices(row["pairwise_distances"], rel2idx)
        by_trial_id[int(row["trial_id"])] = pairs
        if pairs:
            idx = np.fromiter(pairs.keys(), dtype=np.int64, count=len(pairs))
            total[idx] += np.fromiter(pairs.values(), dtype=np.float64, count=len(pairs))
            count[idx] += 1

    retest: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, row in main[main["repeat_of_trial"].notna()].iterrows():
        orig_id = int(row["repeat_of_trial"])
        if orig_id not in by_trial_id:
            continue
        rep = by_trial_id[int(row["trial_id"])]
        orig = by_trial_id[orig_id]
        shared = sorted(set(rep) & set(orig))  # same image set -> same pair keys
        if len(shared) >= 2:
            retest.append((np.array([orig[c] for c in shared]), np.array([rep[c] for c in shared])))

    distances = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
    return PilotSubject(
        participant_id=str(trials["participant_id"].iloc[0]),
        task_version=float(trials["task_version"].iloc[0]),
        n_images=N,
        distances=distances,
        n_obs=count,
        retest_pairs=retest,
        qc_flag_rate=float(main["qc_flag"].astype(bool).mean()) if len(main) else np.nan,
        shine_variant=(
            str(trials["shine_variant"].iloc[0]) if "shine_variant" in trials.columns
            and pd.notna(trials["shine_variant"].iloc[0]) else ""
        ),
    )


def load_pilot_subjects(
        data_dir: str,
        manifest_path: str,
        versions: Optional[Sequence[float]] = None,
        apply_qc: bool = False,
        qc_max_flag_rate: float = 0.30,
        cohorts: Sequence[str] = ("pilot",),
        variants: Optional[Sequence[str]] = None,
) -> List[PilotSubject]:
    """Load completed **pilot** subjects from the flat ``data_dir`` (optionally filtered by version).

    Delegates session/CSV handling and completion filtering to
    ``analysis.utils.parser.load_data``, then reduces each participant's trials to a
    :class:`PilotSubject` on the manifest index space.

    **Production data is excluded by default.** The parser derives ``cohort`` from each file's own
    ``deployment_mode``, and the simulations must not be calibrated on the live study's data: doing so
    would let the sample-size and screening conclusions be shaped by the very cohort they are meant to
    plan, which is circular and amounts to peeking at the running experiment. Pass ``cohorts`` to
    override deliberately (e.g. for a post-hoc check, never for a design decision).

    Note the practical consequence: every v4.0 session is ``production``, so the default pilot-only
    view contains **no screening-block data at all** - reliability is measured from v3.\\* whole-trial
    repeats, which are spread through the session rather than concentrated in a screening block.

    ``versions`` compares against the float ``task_version`` (e.g. ``[3.0]``). ``apply_qc=False`` by
    default; set ``True`` to additionally drop subjects whose ``qc_flag`` rate exceeds
    ``qc_max_flag_rate`` (a robustness check).

    ``variants`` filters on ``shine_variant`` (e.g. ``("pre",)``); ``None`` keeps every variant.
    **Cohort is not a proxy for variant.** The task is documented as serving pilot sessions the
    pre-SHINE images unconditionally, but the data disagrees: of the 47 loadable pilot subjects,
    41 are ``pre`` and 6 are ``post`` (all v3.06). Ground-truth construction must therefore pass
    ``variants=("pre",)`` explicitly - pooling the two variants into one geometry is only masked by
    :func:`_src_to_relpath` collapsing them onto the same manifest index. Noise-model fitting is
    exempt: it estimates a property of subjects rather than of the stimulus set, so it may use both.
    """
    _, rel2idx = load_manifest(manifest_path)
    data = load_data(data_dir)
    participants, trials = data["participants"], data["trials"]
    if trials.empty:
        return []
    keep = participants[participants["cohort"].isin(list(cohorts))]
    if keep.empty:
        return []
    merge_cols = ["participant_id", "task_version"]
    if "shine_variant" in keep.columns:
        merge_cols.append("shine_variant")
    trials = trials.merge(keep[merge_cols], on="participant_id", how="inner")
    subjects: List[PilotSubject] = []
    for _, group in trials.groupby("participant_id", sort=False):
        subj = subject_from_trials(group, rel2idx)
        if versions is not None and subj.task_version not in versions:
            continue
        if variants is not None and subj.shine_variant not in variants:
            continue
        if apply_qc and subj.qc_flag_rate > qc_max_flag_rate:
            continue
        subjects.append(subj)
    return subjects


# --------------------------------------------------------------------------- observables
def within_subject_test_retest(subject: PilotSubject) -> float:
    """Mean Spearman correlation between this subject's repeat trials and their originals.

    NaN if the subject has no (non-degenerate) repeats. This is exactly the statistic the task-v3
    simulation reports as ``subject_test_retest``, so the same target is comparable between the two.
    """
    corrs = [spearmanr(o, r).statistic for o, r in subject.retest_pairs
             if np.ptp(o) > 0 and np.ptp(r) > 0]
    return float(np.nanmean(corrs)) if corrs else np.nan


def _cohort_test_retest(subjects: Sequence[PilotSubject]) -> float:
    """Median within-subject test-retest across a cohort (NaN-subjects ignored)."""
    vals = [within_subject_test_retest(s) for s in subjects]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.median(vals)) if vals else np.nan


def between_subject_agreement(distances: np.ndarray, min_overlap: int = 20) -> dict:
    """Mean dyadic inter-subject Spearman over jointly-observed pairs.

    ``distances`` is a ``(num_subjects, n_pairs)`` array (NaN = unobserved) - works identically for a
    pilot cohort (stack ``subject.distances``) and a simulated cohort
    (``simulate_task_v3_experiment(..., return_per_subject=True)``). For every subject pair it
    correlates the two distance vectors over the pairs *both* observed, keeping only dyads with at
    least ``min_overlap`` shared pairs (each subject sees a different random image subset, so overlap
    is sparse). Returns the mean/SEM agreement plus diagnostics on how many dyads/how much overlap
    backed the estimate.
    """
    S = distances.shape[0]
    observed = ~np.isnan(distances)
    corrs, overlaps = [], []
    for a in range(S):
        for b in range(a + 1, S):
            mask = observed[a] & observed[b]
            k = int(mask.sum())
            if k < min_overlap:
                continue
            va, vb = distances[a, mask], distances[b, mask]
            if np.ptp(va) == 0 or np.ptp(vb) == 0:
                continue
            corrs.append(spearmanr(va, vb).statistic)
            overlaps.append(k)
    return {
        "mean_agreement": float(np.mean(corrs)) if corrs else np.nan,
        "sem_agreement": float(np.std(corrs, ddof=1) / np.sqrt(len(corrs))) if len(corrs) > 1 else np.nan,
        "n_dyads": len(corrs),
        "median_overlap": float(np.median(overlaps)) if overlaps else 0.0,
    }


def stack_distances(subjects: Sequence[PilotSubject]) -> np.ndarray:
    """``(num_subjects, n_pairs)`` matrix of per-subject mean distances (NaN = unobserved)."""
    return np.vstack([s.distances for s in subjects])


# --------------------------------------------------------------------------- aggregate / GT
def pilot_aggregate(subjects: Sequence[PilotSubject]) -> Tuple[np.ndarray, np.ndarray]:
    """Pool subjects into ``(mean_distances, weights)`` ready for ``run_mds``.

    ``mean_distances`` is the per-pair mean over all subjects' observations (0 where unobserved);
    ``weights`` is the matching 0/1 observed mask. **Raises ``RuntimeError`` if the observed-pair
    graph is disconnected** - we never run MDS on a partial graph or silently subset to a component
    (mirrors ``multi_dimensional_scaling.run_mds``'s own guard). A disconnected graph means the pilot
    coverage is insufficient; collect more sessions.
    """
    mean_distances, weights = aggregate_subjects(subjects)
    comps = gt_n_components(weights)
    if comps > 1:
        observed_frac = float(weights.mean())
        raise RuntimeError(
            f"pilot observed-pair graph has {comps} connected components "
            f"(only {observed_frac:.1%} of pairs observed). Refusing to run MDS on a partial graph - "
            "collect more pilot sessions for full connectivity."
        )
    return mean_distances, weights


# --------------------------------------------------------------------------- GT embedding
def build_gt_from_pilot(
        subjects: Sequence[PilotSubject], n_dims: int, method: str = "smacof",
) -> Tuple[np.ndarray, dict]:
    """Pooled-pilot ground-truth coordinates. Thin delegate to :func:`gt_construction.build_gt`.

    ``n_dims`` is now **required**. It used to default to a rule that read a classical-MDS
    eigenspectrum off the *mean-imputed* aggregate and took the smallest dimensionality explaining
    90% of variance, capped at 15. That rule was invalid here: 63.6% of pairs are unobserved and
    were filled with one constant, asserting that all those pairs are equidistant, and k mutually
    equidistant points need k-1 dimensions - so the fill manufactured rank. A synthetic rank-8 space
    put through the same mask and fill reports an effective rank of 193, indistinguishable from the
    real data's 213, and the rule simply returned its cap while carrying no information.

    Choose ``n_dims`` with ``gt_construction.dimensionality_scan`` + ``select_ndim`` instead, which
    measure split-half generalisation and never impute.
    """
    return build_gt(subjects, n_dims, method=method)


# --------------------------------------------------------------------------- calibration
def _simulated_targets(
        gt_embeddings: np.ndarray, noise_scale: float, dispersion: float,
        num_subjects: int, trials_per_subject: int, images_per_trial: int,
        frac_trials_repeated: float, reps: int, seed: int, min_overlap: int,
        noise_df: int = 1, lognormal_sigma: float = 0.0, trial_simulator=None,
) -> Tuple[float, float]:
    """Run the matched simulation and return ``(median_test_retest, mean_between_agreement)``.

    Averaged over ``reps`` independent cohorts for stability (the cohort is only ``num_subjects``).
    ``noise_df`` is the per-subject noise-heterogeneity df (the same value must be used in the sweep);
    ``lognormal_sigma > 0`` instead selects the fitted lognormal noise population (see
    ``noise_population``), which is what the shape fit produces.
    """
    from SpAM_Simulations.models.task_v4_experiment import (
        simulate_task_v4_experiment, TaskV4ExperimentParameters,
    )
    # Routed through task-v4 with the screening block switched off, which is bit-exact to
    # task-v3 (see test_task_v4_experiment.TestEquivalenceToTaskV3) but additionally understands
    # the fitted lognormal noise population. Calibrating the dispersion on the OLD |t(df)| family
    # while the sweep runs on the fitted one would be worse than not recalibrating at all.
    params = TaskV4ExperimentParameters(
        num_subjects=num_subjects, trials_per_subject=trials_per_subject,
        images_per_trial=images_per_trial, subjects_noise_scale=noise_scale,
        subjects_noise_df=noise_df, frac_trials_repeated=frac_trials_repeated,
        perspective_dispersion=dispersion,
        screening_trials=0, screening_repeats=0, screening_min_reliability=-1.0,
        subjects_noise_lognormal_sigma=lognormal_sigma,
    )
    trs, agrs = [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        _, res, per_subject = simulate_task_v4_experiment(
            params, gt_embeddings, rng, verbose=False, return_per_subject=True,
            trial_simulator=trial_simulator,
        )
        trs.append(float(np.nanmedian(res.subject_test_retest)))
        agrs.append(between_subject_agreement(per_subject, min_overlap=min_overlap)["mean_agreement"])
    return float(np.nanmedian(trs)), float(np.nanmean(agrs))


def _fit_1d(target: float, evaluate, grid: np.ndarray) -> float:
    """Pick the grid value whose ``evaluate(x)`` is closest to ``target`` (monotone-agnostic)."""
    vals = np.array([evaluate(x) for x in grid], dtype=np.float64)
    return float(grid[int(np.nanargmin(np.abs(vals - target)))])


def fit_noise_for_test_retest(
        gt_embeddings: np.ndarray,
        target_test_retest: float,
        *,
        noise_df: int,
        lognormal_sigma: float = 0.0,
        images_per_trial: int = 20,
        trials_per_subject: int = 20,
        frac_trials_repeated: float = 0.15,
        num_subjects: int = 20,
        reps: int = 8,
        trial_simulator=None,
        noise_grid: Sequence[float] = tuple(np.round(np.arange(0.1, 3.01, 0.1), 2)),
        seed: int = 0,
) -> Tuple[float, float]:
    """Invert the ``subjects_noise_scale`` (canvas placement noise) that yields a target within-subject
    ``test_retest`` for the given ``noise_df`` and design, returning ``(noise_scale, achieved_tr)``.

    Test-retest is perspective-invariant (a whole-trial repeat re-projects to the same arrangement and
    differs only by fresh placement noise), so the inversion is done at ``dispersion=0`` and is
    independent of ``perspective_dispersion``. It depends on the noise population's SHAPE, so
    ``lognormal_sigma`` must be passed whenever the sweep will use a fitted lognormal population -
    inverting under one family and sweeping under another silently mislabels the whole R axis
    (the mean scale is preserved across families, but the realised reliability is not). It also
    depends on ``noise_df`` (heavy tails shift the
    median) and weakly on ``images_per_trial``; ``num_subjects``/``trials_per_subject`` only affect the
    estimator's variance. If ``target_test_retest`` is outside the grid's achievable range, the closest
    achievable value is returned - inspect ``achieved_tr`` to see how far off it landed.
    """
    def tr(noise: float) -> float:
        return _simulated_targets(
            gt_embeddings, noise, 0.0, num_subjects, trials_per_subject, images_per_trial,
            frac_trials_repeated, reps, seed, min_overlap=25, noise_df=noise_df,
            lognormal_sigma=lognormal_sigma,
            trial_simulator=trial_simulator,
        )[0]
    noise = _fit_1d(target_test_retest, tr, np.asarray(noise_grid))
    return float(noise), float(tr(noise))


def _calibrate(
        gt_embeddings: np.ndarray,
        num_subjects: int,
        target_test_retest: float,
        target_agreement: float,
        *,
        trials_per_subject: int = 20,
        images_per_trial: int = 20,
        frac_trials_repeated: float = 0.15,
        noise_grid: Sequence[float] = tuple(np.round(np.arange(0.1, 3.01, 0.1), 2)),
        dispersion_grid: Sequence[float] = tuple(np.round(np.arange(0.0, 2.01, 0.1), 2)),
        reps: int = 10,
        min_overlap: int = 25,
        noise_df: int = 1,
        seed: int = 0,
) -> dict:
    """Fit ``(subjects_noise_scale, perspective_dispersion)`` so the matched simulation reproduces the
    two pilot targets.

    The task-v3 canvas-placement-noise model makes the two levers a *triangular* system - test-retest
    ``= f(noise)`` (perspective-invariant, since a whole-trial repeat re-projects to the same 2-D
    arrangement and differs only by fresh placement noise), agreement ``= g(noise, dispersion)`` - so a
    sequential fit is exact, not an approximation: (1) fit noise to ``target_test_retest`` at
    ``dispersion=0``; (2) with noise fixed, fit dispersion to ``target_agreement``. ``noise_grid`` is in
    the canvas-ratio units of ``subjects_noise_scale`` - the *mean* jitter/arrangement-spread ratio over
    subjects; because the matched sim uses ``subjects_noise_df=1`` (Cauchy-heavy per-subject spread), the
    typical subject is far less noisy than that mean, so reproducing the pilot's median test-retest lands
    the mean high when ``noise_df`` is small (``df=1`` is Cauchy-heavy). The matched simulation mirrors
    the v3.0 design (``num_subjects`` v3 subjects, 20 trials of 20, 3 repeats) at the given ``noise_df``
    (which must match the sweep's ``subjects_noise_df``, since the fitted noise depends on it);
    ``min_overlap`` is threaded into the simulated between-subject agreement so it is measured over the
    same overlap regime as the pilot target. ``gt_embeddings`` is the pooled-pilot embedding.
    """
    def common(noise, disp):
        return _simulated_targets(
            gt_embeddings, noise, disp, num_subjects, trials_per_subject, images_per_trial,
            frac_trials_repeated, reps, seed, min_overlap, noise_df=noise_df,
            trial_simulator=trial_simulator,
        )

    fitted_noise = _fit_1d(target_test_retest, lambda x: common(x, 0.0)[0], np.asarray(noise_grid))
    fitted_disp = _fit_1d(target_agreement, lambda x: common(fitted_noise, x)[1], np.asarray(dispersion_grid))
    sim_tr, sim_agr = common(fitted_noise, fitted_disp)
    return {
        "subjects_noise_scale": fitted_noise,
        "perspective_dispersion": fitted_disp,
        "subjects_noise_df": noise_df,
        "pilot_test_retest": target_test_retest,
        "pilot_between_agreement": target_agreement,
        "simulated_test_retest": sim_tr,
        "simulated_between_agreement": sim_agr,
        "num_subjects": num_subjects,
    }


def calibrate_params_from_pilot(
        data_dir: str,
        manifest: str,
        *,
        gt_method: str = "smacof",
        reps: int = 10,
        n_dims: Optional[int] = None,
        min_overlap: int = 25,
        noise_df: int = 1,
        gt_coords: Optional[np.ndarray] = None,
        save_gt: Optional[str] = None,
        save_params: Optional[str] = None,
        verbose: bool = True,
) -> Tuple[np.ndarray, dict, dict]:
    """Calibrate the task-v3 simulation to the pilot, end to end; return ``(gt_coords, fit, info)``.

    The single entrypoint shared by the EC2 sweep's calibration flavor (``ec2/run_task_v3_sim.sh`` with
    ``CALIBRATE=true``) and any local/programmatic caller:

    * pool ALL completed pilot subjects -> :func:`build_gt_from_pilot` (weighted SMACOF, or
      ``classical`` for a no-R provisional GT) -> ground-truth coordinates inheriting the real
      spectrum/clusters;
    * **within-subject test-retest** from the *v3.0* subjects (needs the whole-trial repeats) pins the
      noise; **between-subject agreement** from *all* subjects (a population property, not v3-specific -
      more dyads, tighter anchor) pins the perspective dispersion;
    * :func:`_calibrate` -> fitted ``subjects_noise_scale`` + ``perspective_dispersion``.

    ``noise_df`` sets the per-subject noise-heterogeneity df used both here and (necessarily) in the
    sweep; the fitted noise depends on it, so calibrating over several ``noise_df`` values requires one
    call each. Pass ``gt_coords`` to reuse an already-built pilot GT (skips the R/SMACOF build) - e.g.
    when looping ``noise_df`` over one fixed GT. ``save_gt`` / ``save_params`` (if given) persist the GT
    coordinates (``.npy``) and the fitted parameters (``.json``) - the artifacts the downstream sweep
    consumes. Raises ``SystemExit`` if no v3.0 (matched-design) subjects are present. Building the GT
    (``gt_method="smacof"``, ``gt_coords`` not given) requires rpy2.
    """
    allsub = load_pilot_subjects(data_dir, manifest)
    v3 = [s for s in allsub if s.task_version == 3.0]
    if verbose:
        print(f"[load] {len(allsub)} completed sessions; {len(v3)} are v3.0 (matched design)")
    if not v3:
        raise SystemExit("no v3.0 subjects found - the within-subject test-retest target needs the matched design")

    # targets: test-retest is v3-only (whole-trial repeats); agreement pools ALL subjects.
    target_tr = _cohort_test_retest(v3)
    agr = between_subject_agreement(stack_distances(allsub), min_overlap)
    target_agr = agr["mean_agreement"]
    if verbose:
        print(f"[targets] within-subject test-retest (median, v3) = {target_tr:.4f}")
        print(f"[targets] between-subject agreement (all subjects) = {target_agr:.4f} "
              f"(SEM {agr['sem_agreement']:.4f}, {agr['n_dyads']} dyads, median overlap {agr['median_overlap']:.0f})")

    if gt_coords is not None:
        coords = np.asarray(gt_coords, dtype=np.float32)
        info = {"n_dims": coords.shape[1], "method": f"{gt_method} (reused)", "n_subjects": len(allsub),
                "observed_frac": float("nan")}
        if verbose:
            print(f"[gt] reusing supplied GT: N={coords.shape[0]}, n_dims={coords.shape[1]}")
    else:
        if n_dims is None:
            raise ValueError(
                "`n_dims` is required when building a GT. The old default inferred it from the "
                "eigenspectrum of a mean-imputed aggregate, which manufactures dimensionality on "
                "sparse data and simply returned its cap. Choose it with "
                "gt_construction.dimensionality_scan + select_ndim, or pass `gt_coords` to reuse a "
                "GT built that way."
            )
        coords, info = build_gt_from_pilot(allsub, n_dims=n_dims, method=gt_method)
        if verbose:
            print(f"[gt] {info['method']} embedding: N={coords.shape[0]}, n_dims={info['n_dims']}, "
                  f"observed {info['observed_frac']:.1%} of pairs")
            if gt_method == "classical":
                print("[gt] WARNING: provisional no-R ground truth (numpy classical MDS). "
                      "Re-run with gt_method='smacof' for the final numbers.")

    fit = _calibrate(coords, len(v3), target_tr, target_agr, reps=reps, min_overlap=min_overlap,
                     noise_df=noise_df)
    if verbose:
        print(f"[calibrated] noise_df={noise_df}: subjects_noise_scale={fit['subjects_noise_scale']:.3f} "
              f"(sim {fit['simulated_test_retest']:.3f} vs pilot {fit['pilot_test_retest']:.3f}); "
              f"perspective_dispersion={fit['perspective_dispersion']:.3f} "
              f"(sim {fit['simulated_between_agreement']:.3f} vs pilot {fit['pilot_between_agreement']:.3f}); "
              f"n_dims={info['n_dims']}")

    if save_gt:
        np.save(save_gt, coords)
        if verbose:
            print(f"[save] GT coordinates -> {save_gt}  {coords.shape}")
    if save_params:
        with open(save_params, "w", encoding="utf-8") as fh:
            json.dump({**fit, "n_dims": info["n_dims"], "gt_method": info["method"]}, fh, indent=2)
        if verbose:
            print(f"[save] fitted parameters -> {save_params}")
    return coords, fit, info


# --------------------------------------------------------------------------- noise-population fit
def subject_reliability_sample(subjects: Sequence[PilotSubject]) -> np.ndarray:
    """Each subject's mean whole-trial test-retest Spearman, as a 1-D sample (NaN subjects dropped).

    This is the empirical quantity the simulated noise population has to reproduce - not just its
    median (which ``fit_noise_for_test_retest`` already matches) but its whole spread.
    """
    vals = np.array([within_subject_test_retest(s) for s in subjects], dtype=np.float64)
    return vals[~np.isnan(vals)]


def simulate_reliability_sample(
        gt_embeddings: np.ndarray, noise_scale: float, *, family: str = "t", shape: float = 5.0,
        n_subjects: int = 60, n_repeats: int = 4, images_per_trial: int = 20,
        perspective_dispersion: float = 0.2, reps: int = 3, seed: int = 0, trial_simulator=None,
) -> np.ndarray:
    """The simulated counterpart of :func:`subject_reliability_sample`.

    Only repeated trials carry information about test-retest, so each simulated subject runs
    ``n_repeats`` distinct trials plus ``n_repeats`` repeats rather than a full session - the
    reliability distribution is identical and it is an order of magnitude cheaper, which is what
    makes a 2-D grid search affordable.
    """
    from SpAM_Simulations.models.task_v4_experiment import simulate_task_v4_single_subject
    from SpAM_Simulations.models.noise_population import draw_subject_noises
    out = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        noises = draw_subject_noises(n_subjects, noise_scale, rng=rng, family=family, shape=shape)
        for s in range(n_subjects):
            run = simulate_task_v4_single_subject(
                trial_simulator=trial_simulator,
                subject_noise=noises[s], perspective_dispersion=perspective_dispersion,
                t_distinct=n_repeats, k=images_per_trial, n_unique=n_repeats * images_per_trial,
                n_repeats=n_repeats, gt_embeddings=gt_embeddings, rng=rng)
            good = [c for c in run.repeat_correlations if not np.isnan(c)]
            if good:
                out.append(float(np.mean(good)))
    return np.asarray(out)


def fit_noise_population(
        gt_embeddings: np.ndarray, empirical: np.ndarray, *,
        families: Sequence[str] = ("t", "lognormal"),
        t_shapes: Sequence[float] = (2, 3, 5, 8, 15, 30),
        lognormal_shapes: Sequence[float] = (0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0),
        noise_grid: Sequence[float] = tuple(np.round(np.arange(0.4, 2.61, 0.2), 2)),
        n_subjects: int = 60, n_repeats: int = 4, images_per_trial: int = 20,
        perspective_dispersion: float = 0.2, reps: int = 3, seed: int = 0, verbose: bool = True,
        trial_simulator=None,
) -> dict:
    """Jointly fit the noise population's **scale and shape** to an empirical reliability sample.

    ``fit_noise_for_test_retest`` matches only the median reliability, leaving the shape assumed;
    that assumption proved wrong at both tails, and since screening can only truncate a
    distribution its entire yield is set by the shape. This fits both by minimising the
    1-Wasserstein (earth-mover) distance between the simulated and empirical *distributions* of
    per-subject mean reliability.

    Wasserstein rather than a few matched quantiles because the empirical sample is small (tens of
    subjects): it uses every observation, needs no quantile choices, and is in the units of R, so
    the returned ``distance`` is directly interpretable as "the average R-shift between the two
    distributions". Both families are scanned unless ``families`` restricts it - ``"t"`` cannot
    express a cohort more concentrated than a half-normal (CV floor ~0.756), so a fit that lands on
    the largest ``t_shapes`` value is a signal the family is the binding constraint, not the data.

    Returns the best ``{family, shape, noise_scale, distance, cv, simulated_median, ...}`` plus the
    full scanned ``grid`` as a DataFrame, so the fit's sharpness can be inspected rather than
    trusted.
    """
    from scipy.stats import wasserstein_distance
    empirical = np.asarray(empirical, dtype=np.float64)
    empirical = empirical[~np.isnan(empirical)]
    if empirical.size < 5:
        raise ValueError(f"need at least 5 empirical reliabilities to fit a distribution, got {empirical.size}")
    rows = []
    for family in families:
        shapes = t_shapes if family == "t" else lognormal_shapes
        for shape in shapes:
            for scale in noise_grid:
                sim = simulate_reliability_sample(
                    gt_embeddings, float(scale), family=family, shape=float(shape),
                    n_subjects=n_subjects, n_repeats=n_repeats, images_per_trial=images_per_trial,
                    perspective_dispersion=perspective_dispersion, reps=reps, seed=seed,
                    trial_simulator=trial_simulator)
                rows.append(dict(family=family, shape=float(shape), noise_scale=float(scale),
                                 distance=float(wasserstein_distance(sim, empirical)),
                                 sim_median=float(np.median(sim)), sim_mean=float(np.mean(sim)),
                                 sim_q10=float(np.quantile(sim, 0.10)),
                                 sim_q90=float(np.quantile(sim, 0.90))))
            if verbose:
                best = min((r for r in rows if r["family"] == family and r["shape"] == shape),
                           key=lambda r: r["distance"])
                print(f"[fit] {family:<9} shape={shape:<5} -> best scale={best['noise_scale']:.2f} "
                      f"W1={best['distance']:.4f} (median {best['sim_median']:.3f})", flush=True)
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["distance"].idxmin()].to_dict()
    from SpAM_Simulations.models.noise_population import population_cv
    best["cv"] = population_cv(best["family"], best["shape"])
    best["empirical_median"] = float(np.median(empirical))
    best["empirical_n"] = int(empirical.size)
    best["at_shape_boundary"] = bool(
        best["shape"] == max(t_shapes if best["family"] == "t" else lognormal_shapes)
        or best["shape"] == min(t_shapes if best["family"] == "t" else lognormal_shapes))
    # The SCALE boundary, which was missing and is what actually bit. `noise_grid`'s default range
    # was written for the v3/v4 parameterisation, where noise is a ratio to each trial's arrangement
    # spread. Under the canvas it is an absolute fraction of canvas width, and the optimum moves an
    # order of magnitude: reproducing the pilot's median reliability of 0.243 needs ~0.22, while the
    # default grid starts at 0.4. The fit then pins to its own floor and silently reports an
    # achieved median less than half the target, which no other diagnostic here would have caught.
    best["at_noise_boundary"] = bool(
        best["noise_scale"] == max(noise_grid) or best["noise_scale"] == min(noise_grid))
    best["noise_grid_min"] = float(min(noise_grid))
    best["noise_grid_max"] = float(max(noise_grid))
    # How far the best fit actually lands from the target it is trying to match. A large value here
    # means the grid could not reach the data, whatever the family and shape.
    best["median_gap"] = float(best["sim_median"] - best["empirical_median"])
    return {"best": best, "grid": grid}


def fit_dispersion_for_agreement(
        gt_embeddings: np.ndarray, target_agreement: float, *,
        noise_scale: float, noise_df: int = 5, lognormal_sigma: float = 0.0,
        dispersion_grid: Sequence[float] = tuple(np.round(np.arange(0.0, 1.21, 0.05), 2)),
        num_subjects: int = 20, trials_per_subject: int = 20, images_per_trial: int = 20,
        frac_trials_repeated: float = 0.15, reps: int = 5, seed: int = 0, min_overlap: int = 20,
        trial_simulator=None,
) -> Tuple[float, float]:
    """Fit ``perspective_dispersion`` to a target between-subject agreement, noise held fixed.

    Step (2) of the sequential calibration, and it **must be re-run whenever the noise population
    changes** - not only when its mean changes. Between-subject agreement is
    ``g(noise_distribution, dispersion)``: it depends on the whole distribution, because two
    subjects agree less when either is imprecise. Refitting the noise population's *shape* (see
    :func:`fit_noise_population`) therefore moves the agreement curve, and a dispersion calibrated
    against the old shape would be inconsistent with the sweep it feeds.

    Direction, worth knowing before reading the result: a *less* dispersed noise population raises
    agreement at any given dispersion, so matching the same empirical agreement requires a *higher*
    fitted dispersion - which lowers the achievable stability asymptote and raises required-N.

    Returns ``(dispersion, achieved_agreement)``.
    """
    def evaluate(disp):
        return _simulated_targets(
            gt_embeddings, noise_scale, disp, num_subjects, trials_per_subject, images_per_trial,
            frac_trials_repeated, reps, seed, min_overlap, noise_df=noise_df,
            lognormal_sigma=lognormal_sigma,
            trial_simulator=trial_simulator,
        )[1]
    fitted = _fit_1d(target_agreement, evaluate, np.asarray(dispersion_grid))
    return float(fitted), float(evaluate(fitted))
