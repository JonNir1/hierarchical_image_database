#!/usr/bin/env bash
#
# Provision an EC2 instance, run the full SpAM MDS sweep, and upload the results to S3.
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
#   bash run_on_ec2.sh
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

echo ">> workdir=$WORKDIR  n_jobs=$N_JOBS  ref=$GIT_REF  -> $S3_URI"

# --------------------------------------------------------------------------- system packages
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg dirmngr

# Install current R (4.5+) from the CRAN apt repo. Ubuntu 24.04's default r-base is 4.3, which
# is too old for rpy2 3.6 -> at runtime it fails with "undefined symbol: R_getVar" (a symbol
# added in R 4.5). The CRAN repo provides the latest R 4.x for this Ubuntu codename.
curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
  | sudo tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc >/dev/null
APT_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-noble}")"
echo "deb https://cloud.r-project.org/bin/linux/ubuntu ${APT_CODENAME}-cran40/" \
  | sudo tee /etc/apt/sources.list.d/cran.list >/dev/null
sudo apt-get update -y

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git unzip \
  python3 python3-venv python3-dev build-essential gfortran \
  r-base r-base-dev \
  libcurl4-openssl-dev libssl-dev libxml2-dev libblas-dev liblapack-dev \
  libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev \
  libpng-dev libtiff5-dev libjpeg-dev \
  libpcre2-dev liblzma-dev libbz2-dev zlib1g-dev libicu-dev libtirpc-dev
  # ^ last line: the -dev libs rpy2 needs to link its C extension against libR
R --version | head -1   # sanity: should report 4.5.x or newer

# awscli v2 (skip if the AMI already ships it)
if ! command -v aws >/dev/null 2>&1; then
  curl -s "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install
fi

# --------------------------------------------------------------------------- R: smacof
# Start from an empty library so every package matches the installed R version. (Packages built
# for an older R left over from a previous run fail to load under a newer R with errors like
# "undefined symbol: SETLENGTH".)
rm -rf "$R_LIBS_USER"
mkdir -p "$R_LIBS_USER"
# Install from Posit Public Package Manager's PRECOMPILED Ubuntu binaries. Building smacof's
# dependency tree (Hmisc, mice, weights, rmarkdown, ...) from source on a bare instance fails on
# missing system libraries; the binaries sidestep all of that and install in ~1-2 min.
CRAN_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-noble}")"
echo ">> installing R 'smacof' from precompiled binaries ($CRAN_CODENAME)..."
cat > /tmp/install_smacof.R <<RS
options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
  paste(getRversion(), R.version[["platform"]], R.version[["arch"]], R.version[["os"]])))
options(repos = c(P3M = "https://packagemanager.posit.co/cran/__linux__/${CRAN_CODENAME}/latest"))
# Also (re)install Matrix from the same snapshot: R bundles an older Matrix whose ABI can lag
# the one lme4 (a smacof dependency) was built against, which otherwise warns at load time.
install.packages(c("Matrix", "smacof"), lib = Sys.getenv("R_LIBS_USER"))
RS
Rscript --vanilla /tmp/install_smacof.R
Rscript --vanilla -e '.libPaths(Sys.getenv("R_LIBS_USER")); library(smacof); cat("smacof OK\n")'

# --------------------------------------------------------------------------- sparse checkout
# Only the SpAM_Simulations/ package is needed (the sim uses random ground truth, no data/).
rm -rf "$WORKDIR" && mkdir -p "$WORKDIR" && cd "$WORKDIR"
git clone --no-checkout --depth 1 --branch "$GIT_REF" "$REPO_URL" repo
cd repo
git sparse-checkout init --cone
git sparse-checkout set SpAM_Simulations
git checkout "$GIT_REF"

# --------------------------------------------------------------------------- python env
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
# Minimal deps for the sweep (NOT the repo's full requirements.txt, which pulls torch etc.).
pip install -q "numpy>=2.4" "scipy>=1.17" "pandas>=3.0" "scikit-learn>=1.8" \
               "tqdm>=4.67" "joblib>=1.4" rpy2

# --------------------------------------------------------------------------- run the sweep
mkdir -p out
export PYTHONPATH="$PWD"
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

# --------------------------------------------------------------------------- upload to S3
echo ">> uploading results to $S3_URI ..."
aws s3 sync out/        "$S3_URI/out/"        --only-show-errors      # small summary CSVs
aws s3 sync mds_store/  "$S3_URI/mds_store/"  --only-show-errors      # the ~17 GB confdist store

echo ">> ALL DONE. Results at $S3_URI"
echo ">> Tip: you usually only need out/*.csv + mds_store/meta.csv locally; confdists.f32 is the big file."
echo ">> !! TERMINATE THIS EC2 INSTANCE NOW to stop incurring charges !!"
