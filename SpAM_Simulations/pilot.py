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

Session loading and completion filtering are delegated to ``analysis.pilot.parser`` (the canonical
pilot loader: ``load_pilot_data`` returns a demographics-filtered tidy trials frame, and
``parse_pairwise_distances`` parses the per-trial JSON); this module only reduces those trials to the
condensed per-pair distances the calibration needs.

Nothing here writes pilot data or pilot-derived artifacts; ``data/pilot/`` is human-subjects data and
must stay local (never committed). Distances come straight from each trial's ``pairwise_distances``
(already canvas-diagonal-normalised in [0, 1], so comparable across subjects' screen sizes).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from analysis.pilot.parser import load_pilot_data, parse_pairwise_distances
from SpAM_Simulations.experiment import _condensed_pair_indices

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

    def num_observed_pairs(self) -> int:
        return int(np.count_nonzero(self.n_obs))


def _pair_condensed_indices(pairwise_json: str, rel2idx: Dict[str, int]) -> Dict[int, float]:
    """One trial's ``pairwise_distances`` -> ``{condensed_pair_index: distance}`` on the manifest space.

    Reuses ``analysis.pilot.parser.parse_pairwise_distances`` (which returns ``{(src1, src2): dist}``
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
    """Build one :class:`PilotSubject` from a single participant's rows of the parser ``trials`` frame.

    ``trials`` is one participant's slice of ``analysis.pilot.parser.load_pilot_data(...)["trials"]``
    (columns ``pairwise_distances``, ``trial_number``, ``is_trial_repeat``, ``repeat_of_trial_number``,
    ``qc_flag``, ``task_version``, ``participant_id``). Each trial's normalised pairwise distances are
    accumulated into a per-subject condensed sum/count (a verbatim repeat adds a second observation of
    the same pairs, so the stored value is their mean); repeat trials are aligned to their originals
    (via ``repeat_of_trial_number`` in ``trial_number`` space) to give the test-retest pairs.
    """
    N = len(rel2idx)
    n_pairs = N * (N - 1) // 2
    total = np.zeros(n_pairs, dtype=np.float64)
    count = np.zeros(n_pairs, dtype=np.int32)

    by_trial_number: Dict[int, Dict[int, float]] = {}
    for _, row in trials.iterrows():
        pairs = _pair_condensed_indices(row["pairwise_distances"], rel2idx)
        by_trial_number[int(row["trial_number"])] = pairs
        if pairs:
            idx = np.fromiter(pairs.keys(), dtype=np.int64, count=len(pairs))
            total[idx] += np.fromiter(pairs.values(), dtype=np.float64, count=len(pairs))
            count[idx] += 1

    retest: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, row in trials[trials["is_trial_repeat"].astype(bool)].iterrows():
        orig_num = row["repeat_of_trial_number"]
        if pd.isna(orig_num) or int(orig_num) not in by_trial_number:
            continue
        rep = by_trial_number[int(row["trial_number"])]
        orig = by_trial_number[int(orig_num)]
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
        qc_flag_rate=float(trials["qc_flag"].astype(bool).mean()),
    )


def load_pilot_subjects(
        pilot_dir: str,
        manifest_path: str,
        versions: Optional[Sequence[float]] = None,
        apply_qc: bool = False,
        qc_max_flag_rate: float = 0.30,
) -> List[PilotSubject]:
    """Load the completed pilot subjects under ``pilot_dir`` (optionally filtered by ``task_version``).

    Delegates all session/CSV handling and completion filtering to
    ``analysis.pilot.parser.load_pilot_data`` (demographics-aware: consent-revoked and
    erroneous-completion participants are already excluded), then reduces each participant's trials to
    a :class:`PilotSubject` on the manifest index space. ``versions`` compares against the float
    ``task_version`` (e.g. ``[3.0]``). ``apply_qc=False`` by default; set ``True`` to additionally drop
    subjects whose ``qc_flag`` rate exceeds ``qc_max_flag_rate`` (a robustness check).
    """
    _, rel2idx = load_manifest(manifest_path)
    trials = load_pilot_data(pilot_dir)["trials"]
    if trials.empty:
        return []
    subjects: List[PilotSubject] = []
    for _, group in trials.groupby("participant_id", sort=False):
        subj = subject_from_trials(group, rel2idx)
        if versions is not None and subj.task_version not in versions:
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
    if not subjects:
        raise ValueError("no subjects to aggregate")
    n_pairs = subjects[0].distances.shape[0]
    total = np.zeros(n_pairs, dtype=np.float64)
    count = np.zeros(n_pairs, dtype=np.int64)
    for s in subjects:
        obs = s.n_obs > 0
        total[obs] += np.nan_to_num(s.distances[obs]) * s.n_obs[obs]
        count += s.n_obs
    weights = (count > 0).astype(np.float32)
    n_components = connected_components(squareform(weights), directed=False, return_labels=False)
    if n_components > 1:
        observed_frac = float(weights.mean())
        raise RuntimeError(
            f"pilot observed-pair graph has {n_components} connected components "
            f"(only {observed_frac:.1%} of pairs observed). Refusing to run MDS on a partial graph - "
            "collect more pilot sessions for full connectivity."
        )
    mean_distances = np.where(count > 0, total / np.maximum(count, 1), 0.0).astype(np.float32)
    return mean_distances, weights


# --------------------------------------------------------------------------- GT embedding
def _classical_embed(condensed: np.ndarray, ndim: int) -> np.ndarray:
    """Classical-MDS (PCoA) coordinates: double-centre the squared distances, keep top-`ndim`."""
    sq = squareform(condensed).astype(np.float64) ** 2
    n = sq.shape[0]
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ sq @ centring
    vals, vecs = np.linalg.eigh(gram)
    idx = np.argsort(vals)[::-1][:ndim]
    return (vecs[:, idx] * np.sqrt(np.clip(vals[idx], 0, None))).astype(np.float32)


def _choose_n_dims(eigenvalues: np.ndarray, var_threshold: float = 0.9, cap: int = 15) -> int:
    """Smallest dimensionality whose positive eigenvalues explain >= `var_threshold` (capped)."""
    pos = eigenvalues[eigenvalues > 0]
    cum = np.cumsum(pos) / pos.sum()
    return int(min(cap, np.searchsorted(cum, var_threshold) + 1))


def build_gt_from_pilot(
        subjects: Sequence[PilotSubject], n_dims: Optional[int] = None, method: str = "smacof",
) -> Tuple[np.ndarray, dict]:
    """Pooled-pilot ground-truth coordinates for the calibration simulation.

    Aggregates all subjects (raises on a disconnected graph), reads the eigenspectrum to pick
    ``n_dims`` (if not given), and embeds:

    * ``method="smacof"`` - weighted SMACOF via ``multi_dimensional_scaling.run_mds`` (needs R/rpy2;
      uses the 0/1 weights so unobserved pairs don't bias the fit). The canonical path.
    * ``method="classical"`` - numpy classical MDS on the mean-imputed aggregate (no R). A
      **provisional** path for environments without R; the spectrum head matches, the tail is rougher.

    Returns ``(coords[N, n_dims] float32, info)``.
    """
    from SpAM_Simulations.metrics import classical_mds_eigenvalues
    dists, weights = pilot_aggregate(subjects)  # raises if disconnected
    imputed = dists.copy()
    imputed[weights == 0] = dists[weights > 0].mean()
    eig = classical_mds_eigenvalues(imputed)
    if n_dims is None:
        n_dims = _choose_n_dims(eig)
    if method == "smacof":
        from SpAM_Simulations.multi_dimensional_scaling import run_mds  # lazy: imports R
        out = run_mds(dists=dists, weights=weights, ndim=n_dims)
        coords = np.asarray(out["conf"], dtype=np.float32)
    elif method == "classical":
        coords = _classical_embed(imputed, n_dims)
    else:
        raise ValueError(f"method must be 'smacof' or 'classical', got {method!r}")
    info = {"n_dims": n_dims, "method": method, "n_subjects": len(subjects),
            "observed_frac": float(weights.mean()), "eigenvalues": eig[:n_dims + 5]}
    return coords, info


# --------------------------------------------------------------------------- calibration
def _simulated_targets(
        gt_embeddings: np.ndarray, noise_scale: float, dispersion: float,
        num_subjects: int, trials_per_subject: int, images_per_trial: int,
        frac_trials_repeated: float, reps: int, seed: int, min_overlap: int,
) -> Tuple[float, float]:
    """Run the matched simulation and return ``(median_test_retest, mean_between_agreement)``.

    Averaged over ``reps`` independent cohorts for stability (the cohort is only ``num_subjects``).
    """
    from SpAM_Simulations.task_v3_experiment import (
        simulate_task_v3_experiment, TaskV3ExperimentParameters,
    )
    params = TaskV3ExperimentParameters(
        num_subjects=num_subjects, trials_per_subject=trials_per_subject,
        images_per_trial=images_per_trial, subjects_noise_scale=noise_scale,
        subjects_noise_df=1, frac_trials_repeated=frac_trials_repeated,
        perspective_dispersion=dispersion,
    )
    trs, agrs = [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        _, res, per_subject = simulate_task_v3_experiment(
            params, gt_embeddings, rng, verbose=False, return_per_subject=True
        )
        trs.append(float(np.nanmedian(res.subject_test_retest)))
        agrs.append(between_subject_agreement(per_subject, min_overlap=min_overlap)["mean_agreement"])
    return float(np.nanmedian(trs)), float(np.nanmean(agrs))


def _fit_1d(target: float, evaluate, grid: np.ndarray) -> float:
    """Pick the grid value whose ``evaluate(x)`` is closest to ``target`` (monotone-agnostic)."""
    vals = np.array([evaluate(x) for x in grid], dtype=np.float64)
    return float(grid[int(np.nanargmin(np.abs(vals - target)))])


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
    the mean around ~2 (grid runs to 3.0). The
    matched simulation mirrors the v3.0 design (``num_subjects`` v3 subjects, 20 trials of 20,
    3 repeats); ``min_overlap`` is threaded into the simulated between-subject agreement so it is
    measured over the same overlap regime as the pilot target. ``gt_embeddings`` is the pooled-pilot
    embedding.
    """
    def common(noise, disp):
        return _simulated_targets(
            gt_embeddings, noise, disp, num_subjects, trials_per_subject, images_per_trial,
            frac_trials_repeated, reps, seed, min_overlap,
        )

    fitted_noise = _fit_1d(target_test_retest, lambda x: common(x, 0.0)[0], np.asarray(noise_grid))
    fitted_disp = _fit_1d(target_agreement, lambda x: common(fitted_noise, x)[1], np.asarray(dispersion_grid))
    sim_tr, sim_agr = common(fitted_noise, fitted_disp)
    return {
        "subjects_noise_scale": fitted_noise,
        "perspective_dispersion": fitted_disp,
        "subjects_noise_df": 1,
        "pilot_test_retest": target_test_retest,
        "pilot_between_agreement": target_agreement,
        "simulated_test_retest": sim_tr,
        "simulated_between_agreement": sim_agr,
        "num_subjects": num_subjects,
    }


def calibrate_params_from_pilot(
        pilot_dir: str,
        manifest: str,
        *,
        gt_method: str = "smacof",
        reps: int = 10,
        n_dims: Optional[int] = None,
        min_overlap: int = 25,
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

    ``save_gt`` / ``save_params`` (if given) persist the GT coordinates (``.npy``) and the fitted
    parameters (``.json``) - the artifacts the downstream sweep consumes. Raises ``SystemExit`` if no
    v3.0 (matched-design) subjects are present. Steps needing R (``gt_method="smacof"``) require rpy2.
    """
    allsub = load_pilot_subjects(pilot_dir, manifest)
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

    coords, info = build_gt_from_pilot(allsub, n_dims=n_dims, method=gt_method)
    if verbose:
        print(f"[gt] {info['method']} embedding: N={coords.shape[0]}, n_dims={info['n_dims']}, "
              f"observed {info['observed_frac']:.1%} of pairs")
        if gt_method == "classical":
            print("[gt] WARNING: provisional no-R ground truth (numpy classical MDS). "
                  "Re-run with gt_method='smacof' for the final numbers.")

    fit = _calibrate(coords, len(v3), target_tr, target_agr, reps=reps, min_overlap=min_overlap)
    if verbose:
        print(f"[calibrated] subjects_noise_scale={fit['subjects_noise_scale']:.3f} "
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
