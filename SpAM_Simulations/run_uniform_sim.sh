#!/usr/bin/env bash
#
# Provision an EC2 instance, run the full SpAM MDS sweep for the UNIFORM (original) simulation
# - every trial draws images_per_trial images uniformly at random from the whole pool, no
# per-subject trial design - and upload the results to S3.
#
# For the realistic, per-subject trial-design simulation, see the sibling run_realistic_sim.sh.
# Both source the shared provisioning logic in prepare_machine.sh (must be copied alongside
# this script - see README.md's "Running on EC2" section for the full allocate-to-terminate
# cookbook).
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
#   export S3_URI=s3://<your-bucket>/spam-mds/run-$(date +%Y%m%d)
#   bash run_uniform_sim.sh
#
# Pick an instance with many vCPUs (the sweep is embarrassingly parallel), e.g. c7i.4xlarge
# (16 vCPU). REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-mds/run1}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
N_JOBS="${N_JOBS:-$(nproc)}"            # MDS worker processes (default: all vCPUs)
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# --------------------------------------------------------------------------- run the sweep
N_JOBS="$N_JOBS" python - <<'PY'
import os
from SpAM_Simulations.config import SimulationConfig, MDSSweepConfig
from SpAM_Simulations import pipeline

cfg = SimulationConfig(
    n_images=754, n_dims=10,
    num_subjects=[20, 30, 50, 75, 250],
    trials_per_subject=[8, 10, 15],
    images_per_trial=[16, 20, 25],
    subjects_noise_scale=[0.0, 0.2, 0.5, 0.8],
    subjects_noise_df=[1, 5],
    reps=5, seed=42,
)
sim = pipeline.generate_simulation(cfg, verbose=True)
pipeline.compute_coverage_table(sim).to_csv("out/coverage.csv", index=False)
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
