#!/usr/bin/env bash
#
# Provision an EC2 instance, run the full SpAM MDS sweep for the TASK-V3 simulation - the
# generative, coordinate-space observation model (per-subject PC "perspective" + a local 2-D
# arrangement projection per trial, replacing the additive-distance-noise model of v0.1/v2.3/v2.4)
# - and upload the results to S3.
#
# Task v3.0 (SpAM_Task/js/trial_generator.js) drops the frac_images_repeated cross-context lever:
# every image appears in exactly one distinct trial, and the only repetition is a verbatim
# whole-trial repeat (frac_trials_repeated, giving the within-subject test-retest reliability).
# This sweep therefore has NO frac_images_repeated, and adds two new levers:
#   * perspective_dispersion - between-subject disagreement (spread of each subject's per-PC
#     weights). 0 = everyone shares the ground-truth geometry (pure-noise model).
#   * use_isotropic (set per run below, not swept) - the ground-truth eigenvalue spectrum:
#     True  = isotropic, the conservative full-rank "every dimension equally important" bound;
#     False = geometric `decay` spectrum (+ optional hierarchical `n_clusters`), the realistic
#     case where low-variance dims rarely surface in any trial's 2-D slice. Run the sweep twice
#     (use_isotropic True and False) to bracket the required-N estimate.
#
# Sibling scripts: run_task_v2_4_sim.sh (additive-noise model + both repeat levers),
# run_task_v2_3_sim.sh, run_task_v0_1_sim.sh. All source the shared prepare_machine.sh (must be
# copied alongside - see README.md's "Running on EC2" cookbook).
#
# Target: Ubuntu 22.04/24.04 (apt + CRAN). On Amazon Linux swap the apt blocks for dnf.
#
# Prerequisites
#   * The commit/branch you want to run is PUSHED to the remote (this clones from it).
#   * The instance can reach the repo (public repo, or a PAT/deploy key in REPO_URL/ssh).
#   * S3 access: attach an IAM role with s3:PutObject on the bucket (preferred), or run
#     `aws configure` with the project IAM user's credentials before this script.
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/task-v3-isotropic
#   export USE_ISOTROPIC=true            # or false for the realistic anisotropic spectrum
#   bash run_task_v3_sim.sh
#
# This grid: 4 num_subjects x 3 trials_per_subject x 1 images_per_trial x 2 noise_scale x 1
# noise_df x 4 frac_trials_repeated x 3 perspective_dispersion = 288 combos x 5 reps x 9 target
# ndims = ~12960 MDS fits. c7i.4xlarge (16 vCPU) is fine but the sweep is ~3x the v2.4 one.
# REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/task-v3-isotropic}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
USE_ISOTROPIC="${USE_ISOTROPIC:-false}"   # ground-truth spectrum: true=isotropic bound, false=realistic
# MDS worker processes. All vCPUs OOM'd a run (each worker holds its own R/smacof process), so
# default to 2/3 of them.
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# --------------------------------------------------------------------------- run the sweep
N_JOBS="$N_JOBS" USE_ISOTROPIC="$USE_ISOTROPIC" python - <<'PY'
import os
from SpAM_Simulations.config import TaskV3SimulationConfig, MDSSweepConfig
from SpAM_Simulations import pipeline

use_isotropic = os.environ["USE_ISOTROPIC"].strip().lower() in ("1", "true", "yes")
cfg = TaskV3SimulationConfig(
    n_images=725, n_dims=10,
    num_subjects=[30, 50, 75, 250],
    trials_per_subject=[10, 15, 20],
    images_per_trial=[20],
    subjects_noise_scale=[0.5, 0.8],
    subjects_noise_df=[1],
    frac_trials_repeated=[0.0, 0.1, 0.15, 0.2],
    perspective_dispersion=[0.0, 0.1, 0.25],
    use_isotropic=use_isotropic,         # set per run (see header); False also uses decay/n_clusters
    decay=0.7,
    n_clusters=None,
    reps=5, seed=42,
)
print(f"ground-truth spectrum: {'isotropic (conservative bound)' if use_isotropic else 'anisotropic (realistic)'}")
sim = pipeline.generate_task_v3_simulation(cfg, verbose=True)
pipeline.compute_coverage_table(sim).to_csv("out/coverage.csv", index=False)   # test-retest cols included automatically
pipeline.compute_stability_table(sim).to_csv("out/stability.csv", index=False)

sweep = MDSSweepConfig(min_ndim=2, max_iters=500, convergence_tol=1e-6, precalc_init=False)
store = pipeline.run_mds_sweep(
    sim, sweep, "mds_store",
    parallel=True, n_jobs=int(os.environ["N_JOBS"]), verbose=True,
)
pipeline.compute_embedding_stability(store).to_csv("out/embedding_stability.csv", index=False)
print(f"done: {len(store)} MDS results")
PY

upload_and_finish
