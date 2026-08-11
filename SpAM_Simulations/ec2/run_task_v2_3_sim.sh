#!/usr/bin/env bash
#
# Provision an EC2 instance, run the full SpAM MDS sweep for the TASK-V2.3 simulation - each
# subject draws their own per-subject image subset and trial allocation (matching SpAM_Task's
# actual design, including controlled image repetition via frac_images_repeated) - and upload
# the results to S3.
#
# For the original, task-v0.1 simulation, see the sibling run_task_v0_1_sim.sh. Both source
# the shared provisioning logic in prepare_machine.sh (must be copied alongside this script -
# see README.md's "Running on EC2" section for the full allocate-to-terminate cookbook).
#
# Target: Ubuntu 22.04/24.04 (apt + CRAN). On Amazon Linux swap the apt blocks for dnf and
# install R from the amazon-linux-extras / EPEL repos; everything else is identical.
#
# Prerequisites
#   * The commit/branch you want to run is PUSHED to the remote (this clones from it).
#   * The instance can reach the repo (public repo, or a PAT/deploy key in REPO_URL/ssh).
#   * S3 access: attach an IAM role with s3:PutObject on the bucket to the instance (preferred),
#     or run `aws configure` with the project IAM user's credentials before this script.
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/task-v2.3
#   bash run_task_v2_3_sim.sh
#
# This grid is actually lighter than run_task_v0_1_sim.sh's: 72 parameter combinations x 5 reps x
# 9 target ndims = ~3240 MDS fits, vs. ~16200 for the task-v0.1 script's larger grid. Per-fit cost
# is comparable between the two (similar n_images, same ndim range/max_iters), so c7i.4xlarge
# (16 vCPU) is plenty here too. REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/run1}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
# MDS worker processes. All vCPUs OOM'd/SEGFAULT'd a run (each worker holds its own R/smacof
# process, and the combined working set outgrew available memory) - default to 2/3 of them.
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# --------------------------------------------------------------------------- run the sweep
N_JOBS="$N_JOBS" python - <<'PY'
import os
from SpAM_Simulations.core.config import TaskV2_3SimulationConfig, MDSSweepConfig
from SpAM_Simulations.core import pipeline

cfg = TaskV2_3SimulationConfig(
    n_images=725, n_dims=10,
    num_subjects=[30, 50, 75, 250],
    trials_per_subject=[10, 15, 20],
    images_per_trial=[20],
    subjects_noise_scale=[0.5, 0.8],
    subjects_noise_df=[1],
    frac_images_repeated=[1 / 3, 1 / 7, 0.0],
    reps=5, seed=42,
)
sim = pipeline.generate_task_v2_3_simulation(cfg, verbose=True)
pipeline.compute_coverage_table(sim).to_csv("out/coverage.csv", index=False)   # SNR cols included automatically
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
