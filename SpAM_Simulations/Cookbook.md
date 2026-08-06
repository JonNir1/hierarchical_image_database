# SpAM_Simulations Cookbook

Operational recipes: how to run things locally, how to get R working, and how to drive a full-scale
sweep on EC2. For *what* the simulations answer and *how* the pipeline is put together, see
[README.md](README.md).

## Contents
- [Quick start](#quick-start)
- [Performance and storage notes](#performance-and-storage-notes)
- [Running with R (rpy2 + smacof)](#running-with-r-rpy2--smacof)
- [Running on EC2](#running-on-ec2)
  - [Calibrated flavor (fit to pilot first)](#calibrated-flavor-fit-to-pilot-first)
  - [Task-v4 flavor (screening block, pilot-calibrated)](#task-v4-flavor-screening-block-pilot-calibrated)
  - [Two-stage flavor: GT construction, then designed vs random](#two-stage-flavor-gt-construction-then-designed-vs-random)

## Quick start

`pipeline.generate_simulation` and the coverage/stability tables need no R; only
`run_mds_sweep` (the weighted-SMACOF reconstruction step) requires R + `smacof`.

```python
from SpAM_Simulations.config import SimulationConfig, MDSSweepConfig
from SpAM_Simulations import pipeline

cfg = SimulationConfig(
    n_images=60, n_dims=8,                       # or: gt_embeddings=<N x D array> for real data
    num_subjects=[20, 40], trials_per_subject=[10], images_per_trial=[16],
    subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1], reps=3, seed=42,
)
sim = pipeline.generate_simulation(cfg)                      # bit-exact, reproducible from seed
coverage  = pipeline.compute_coverage_table(sim)            # no R needed
stability = pipeline.compute_stability_table(sim)           # no R needed

sweep = MDSSweepConfig(ndims=[6, 8], max_iters=300, precalc_init=False)
store = pipeline.run_mds_sweep(sim, sweep, "mds_store", parallel=False)   # needs R; resumable
emb   = pipeline.compute_embedding_stability(store)         # Spearman of distance vectors (higher=better)
gen   = pipeline.compute_embedding_generalizability(store)  # Procrustes M^2 of the spaces (LOWER=better)
```

Or run the bundled example: `python -m SpAM_Simulations.example_pipeline` (from the repo root).

## Performance and storage notes
- Generation is vectorized and runs in condensed form (~9-10x faster than the original loop,
  ~half the memory). The serial path is bit-exact and fully reproducible from `seed`.
- `run_mds_sweep(parallel=True, n_jobs=...)` distributes the independent MDS runs across
  processes (joblib/loky), streaming results to disk so peak memory stays bounded.
- `ResultStore` keeps a human-readable `meta.csv` plus flat float32 binaries (memory-mapped on
  read), replacing the old multi-GB append-only pickle: `confdists.f32` holds each fit's condensed
  distance vector and `confs.f32` its `(n_images, ndim)` coordinates, zero-padded to `max_ndim`.
- **`confdists.f32` is not uploaded to S3.** It is exactly `pdist(conf)` per row, so it is fully
  recoverable from `confs.f32` at roughly a twentieth of the size. `upload_and_finish` excludes it;
  set `UPLOAD_CONFDISTS=true` to override. A store written without configurations
  (`store_conf=False`) has nothing to recover from, so there the exclusion is skipped automatically
  and a warning is printed. Syncing it unconditionally is what accumulated ~26 GiB of pure
  redundancy across earlier runs.

For real pilot data instead of synthetic, see
[Calibrating to pilot data](README.md#step-a-calibrate-to-the-pilot) in the README. For
large-scale sweeps, see [Running on EC2](#running-on-ec2) below.

## Running with R (rpy2 + smacof)
`multi_dimensional_scaling.py` imports R at load time. R 4.5 + the `smacof` package are
required. On Windows without Rtools, `R CMD config` cannot run; rpy2 must fall back to the
DLL in `R_HOME/bin/x64`. That fallback only triggers when the `config` subprocess fails
*cleanly*, which it does from a normal Windows shell but **not** from Git Bash (where `sh`
makes the config script exit 0 with empty output and crashes rpy2's parser).

Working setup used for this project:

```
R_HOME       = C:\Program Files\R\R-4.5.2
R_LIBS_USER  = C:\Users\nirjo\R_library\4.5      # where smacof is installed
```

From a plain PowerShell/cmd session (with the venv active) rpy2 initialises fine. If you must
run it under Git Bash, give the Python process a Windows-only PATH (no Unix `sh`), e.g.
`PATH="/c/Program Files/R/R-4.5.2/bin/x64:/c/Windows/System32:/c/Windows"`, so `R CMD config`
fails fast and rpy2 falls back to `bin/x64`.

`multi_dimensional_scaling.py` automatically prepends `R_HOME\bin\x64` to PATH at import time, so
loading R packages (whose DLLs depend on `R.dll`/BLAS there) works without you editing PATH. If R
still isn't found, set `R_HOME` explicitly.

## Running on EC2

Eight shell scripts, under `ec2/`, handle the full-scale sweeps remotely. The last two are the
current programme; the rest are the historical runs, kept because their results are cited.
- `ec2/prepare_machine.sh` - shared provisioning (system packages, R 4.5 + `smacof`, awscli v2,
  sparse-checkout clone of `SpAM_Simulations/`, Python venv), plus `stage_pull`/`stage_push` for
  handing artifacts between stages and `upload_and_finish` for the final sync. Sourced, not run
  directly.
- `ec2/run_task_v0_1_sim.sh` - runs the task-v0.1 (original) simulation's full-study sweep.
- `ec2/run_task_v2_3_sim.sh` - runs the task-v2.3 (per-subject trial design) simulation's
  full-study sweep (actually fewer total MDS fits than the task-v0.1 sweep - same instance
  type is fine for both).
- `ec2/run_task_v2_4_sim.sh` - runs the task-v2.4 simulation's full-study sweep (task-v2.3
  design plus the `frac_trials_repeated` whole-trial-repeat lever / test-retest reliability).
  Its grid fixes `frac_images_repeated=0.0` and sweeps `frac_trials_repeated`, because the two
  levers compete for the same trials: a repeat may only duplicate a singles-only trial, so at
  `k=20` any `frac_images_repeated>0` saturates every trial with doubled images and leaves none
  to repeat (matching the deployed `task_config.json`). The doubled-image SNR is characterised
  by the task-v2.3 sweep instead.
- `ec2/run_task_v3_sim.sh` - runs the task-v3 (generative coordinate-space model) sweep. Two flavors:
  the default synthetic-GT swept run, and (with `CALIBRATE=true`) a pilot-calibrated run - see
  *Calibrated flavor* below.
- `ec2/run_task_v4_sim.sh` - runs the task-v4 (v3 + screening block) sweep. **Pilot-calibrated
  only** - v4 exists to produce a calibrated number, so the synthetic flavor is not duplicated. The
  design is fixed to the deployed `task_config.json` (20 images/trial; 8 screening + 14
  experimental trials) and the sweep covers `num_subjects` x `screening_min_reliability` x target
  test-retest R. One run answers four questions: embedding generalizability between two cohorts of
  N, whether screening lowers required-N and at what recruitment cost, required-N at the deployed
  design, and item-level recoverability. See *Task-v4 flavor* below.
- `ec2/run_gt_construction.sh` - **stage 1 of the current programme.** Chooses the GT
  dimensionality by split-half agreement, corroborates it with leave-k-out CV over subjects, and
  fits the final embedding on the 41 pre-SHINE pilot subjects. Its `gt/selection.json` supplies
  `N_DIMS` to every later script. See *Two-stage flavor* below.
- `ec2/run_design_comparison.sh` - **stage 2 of the current programme.** Designed versus random
  image-to-trial allocation, both arms in one store as a swept `allocation_mode` lever. Pulls
  stage 1's ground truth and never rebuilds it. See *Two-stage flavor* below.

All entrypoints `source` `prepare_machine.sh` by relative path, so copy `prepare_machine.sh`
together with whichever entrypoint(s) you want onto the instance (don't transfer an entrypoint
alone).

### Cookbook: allocate -> run -> verify -> terminate

Steps (0)-(4) get you to an SSH prompt on a fresh instance; then tmux a sweep script (or, for the
pilot-calibrated run, follow *Calibrated flavor* below).

Known infra for this project: security group `sg-0e1f88c3d550f7154`, key pair `paf-key`
(`.pem` kept outside the repo), IAM instance profile `spam-simulations` (has S3 access).

**(0) One-time setup constants** (PowerShell):
```powershell
$S3_URI        = "s3://jon-nir/spam-simulations/task-v3"
$SG_ID         = "sg-0e1f88c3d550f7154"
$KEY_NAME      = "paf-key"
$KEY_PATH      = "C:\Users\nirjo\Documents\projects\__secrets__\paf-key.pem"
$IAM_PROFILE   = "spam-simulations"
$INSTANCE_TYPE = "c7i.4xlarge"     # 16 vCPU
```
If your ISP-assigned IP changed since the security group rule was added:
```powershell
$MY_IP = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com").Trim()
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr "$MY_IP/32"
```

**(1) Find the latest Ubuntu AMI** via Canonical's official SSM parameter (no hardcoded AMI ID,
no need to know Canonical's account ID, and no relying on AMI *naming* conventions - just the
documented "current AMI" pointer Canonical/AWS maintain for this purpose):
```powershell
$UBUNTU_VERSION = "24.04"     # or "22.04" - matches ec2/prepare_machine.sh's own codename fallback (noble)
$AMI_ID = aws ssm get-parameters `
  --names "/aws/service/canonical/ubuntu/server/$UBUNTU_VERSION/stable/current/amd64/hvm/ebs-gp3/ami-id" `
  --query "Parameters[0].Value" --output text
```

**(2) Allocate the machine (Spot) and get its IP:**
```powershell
$INSTANCE_ID = aws ec2 run-instances `
  --image-id $AMI_ID `
  --instance-type $INSTANCE_TYPE `
  --key-name $KEY_NAME `
  --security-group-ids $SG_ID `
  --iam-instance-profile Name=$IAM_PROFILE `
  --instance-market-options 'MarketType=spot' `   # <- DROP this line for On-Demand
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100}' `
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spam-ec2}]' `
  --query "Instances[0].InstanceId" --output text

aws ec2 wait instance-running --instance-ids $INSTANCE_ID
$IP = aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text
$IP
```
(100 GB root volume - the task-v2.3 sweep's `mds_store/` is larger than the task-v0.1 sweep's
~17 GB; shrink back down for task-v0.1-only runs if you want to save a few cents.)

`--instance-market-options`/`--block-device-mappings` use AWS CLI's shorthand syntax
(`key=value`, no JSON) rather than raw `'{"...":"..."}'` - PowerShell strips embedded double
quotes when handing an argument to a native executable like `aws.exe`, so inline JSON like
`'[{"DeviceName":"/dev/sda1",...}]'` arrives at the CLI with every `"` silently removed and
fails with a JSON parse error. Shorthand syntax has no `"` characters to mangle.

**(3) Transfer the scripts:**
```powershell
cd C:\Users\nirjo\Documents\University\PhD\Projects\hierarchical_image_database
scp -i $KEY_PATH SpAM_Simulations\ec2\*.sh ubuntu@${IP}:~   # copy all EC2 scripts to home dir
```

**(4) SSH in:**
```powershell
ssh -i $KEY_PATH ubuntu@$IP
```

**(5) On the instance - set the run-specific constants:**
```bash
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/task-v0.1   # or .../task-v2.3, .../task-v2.4, matching the script you run in step (6)
# WORKDIR, N_JOBS, R_LIBS_USER all have sane defaults (see each script's header) - override only if needed
```

**(6) Start tmux, run the sweep, log it:**
```bash
tmux new -s spam
bash run_task_v0_1_sim.sh 2>&1 | tee run.log
# or: bash run_task_v2_3_sim.sh 2>&1 | tee run.log
# or: bash run_task_v2_4_sim.sh 2>&1 | tee run.log
```

**(7) Detach / re-attach** (safe to close the SSH session after detaching - the script keeps
running under tmux):
```bash
# detach:   Ctrl-b d
ssh -i $KEY_PATH ubuntu@$IP    # (from a new PowerShell window, if you closed the first one)
tmux attach -t spam
```

**(8) Monitor memory** (from a second SSH session, or split the tmux pane with `Ctrl-b %`):
```bash
watch -n5 free -h
```
For a quick peek without grabbing the tmux session, `tail -f run.log` works too.

**(9) Verify the upload landed on S3** (from the instance or locally once you have `S3_URI`):
```bash
aws s3 ls $S3_URI/out/
aws s3 ls $S3_URI/mds_store/
```

**(10) Manual upload to S3** (if the automatic upload in step (6) didn't run or didn't finish -
e.g. the script crashed before reaching `upload_and_finish`):
```bash
cd ~/spam_run/repo   # default WORKDIR/repo - adjust if you overrode WORKDIR in step (5)
export S3_URI=s3://jon-nir/spam-simulations/task-v0.1   # or .../task-v2.3 - re-set if this is a new session
aws s3 sync out/       "$S3_URI/out/"       --only-show-errors
aws s3 sync mds_store/ "$S3_URI/mds_store/" --only-show-errors

ls -la out/                                                       # coverage.csv, stability.csv, embedding_stability.csv
aws s3 ls "$S3_URI/out/"                      # confirm all 3 are in S3
aws s3 ls "$S3_URI/mds_store/"      # confirm confdists.f32 + meta.csv are in S3
```

**(11) Terminate and confirm:**
```powershell
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID
aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].State.Name" --output text
# expect: terminated
```

**(12) Download the results to your local PC** for analysis with `evaluate_simulation.ipynb`.
The notebook reads runs from `SpAM_Simulations/sim_results/<run-name>/`, where `<run-name>` is
the `S3_URI`'s last path segment (`task-v0.1` / `task-v2.3` / `task-v2.4`). Run from the repo
root (PowerShell):
```powershell
cd C:\Users\nirjo\Documents\University\PhD\Projects\hierarchical_image_database
$RUN_NAME = "task-v2.4"                                   # match the S3_URI you uploaded to
$S3_URI   = "s3://jon-nir/spam-simulations/$RUN_NAME"
$DEST     = "SpAM_Simulations\sim_results\$RUN_NAME"

# Pull out/*.csv, mds_store/meta.csv, store_info.json AND confs.f32 - but never confdists.f32.
aws s3 sync "$S3_URI/out/"       "$DEST\out\"       --only-show-errors
aws s3 sync "$S3_URI/mds_store/" "$DEST\mds_store\" --exclude "confdists.f32" --only-show-errors

Get-ChildItem -Recurse $DEST | Select-Object FullName
```

**`confs.f32` must come down; `confdists.f32` must not.** The configurations are what
`compute_embedding_generalizability`, `compute_item_generalizability` and the entire cluster
analysis read - the last of these recomputes every distance as `pdist(store.conf(row, ndim))`,
which is bit-identical to the stored distance row and additionally makes Ward linkage valid (Ward's
objective is defined only for Euclidean input). A conf row is `n_images * max_ndim` floats against
`n_images^2 / 2` for a distance row, so for a 480-fit sweep that is ~28 MB rather than ~500 MB.

Since the confdists-excluding sync is now the *normal* download, `ResultStore.open` derives its
record count from `meta.csv` rather than from the `confdists.f32` file size, and a conf-only store
opens without complaint; `store.has_confdists` reports which kind you have and `store.confdist(row)`
raises an error naming the missing file. **Newer runs do not upload `confdists.f32` at all** (see
[Performance and storage notes](#performance-and-storage-notes)), so for those the `--exclude` is
belt-and-braces rather than load-bearing.
`sim_results/` is gitignored, so these stay local. Then open
`SpAM_Simulations/evaluate_simulation.ipynb` and set `RUN_RESULTS_DIR = "sim_results/task-v2.4"`.

### Calibrated flavor (fit to pilot first)

`run_task_v3_sim.sh` with `CALIBRATE=true` runs a short (~1-5 min) prelude that fits the simulation to
the real pilot, then runs the **same** convergence sweep with the pilot GT + fitted
`subjects_noise_scale`/`perspective_dispersion` (see
[Calibrating to pilot data](README.md#step-a-calibrate-to-the-pilot) in the README for the method).
Everything for the study lives under one **private** `S3_URI` prefix, the data path mirroring the
local flat `data/`:
```
$S3_URI/data/         <- INPUT  you stage once: session + demographics CSVs + stimuli_manifest.json
                         (flat, both cohorts; the calibration itself uses only the pilot ones)
$S3_URI/calibration/  <- OUTPUT calibrate.log, calibrated_params.json, gt_pilot_coords.npy
$S3_URI/out/  $S3_URI/mds_store/   <- OUTPUT of the convergence sweep
```

**Stage the pilot once** (the CSVs + manifest are gitignored, so never in the repo clone - the
entrypoint pulls them from S3 and deletes them from the box at exit):
```powershell
$S3_URI = "s3://jon-nir/spam-simulations/task-v3"
aws s3 cp data/                          "$S3_URI/data/" --recursive --exclude "*.pdf"   # session + demographics CSVs
aws s3 cp SpAM_Task/stimuli_manifest.json "$S3_URI/data/"                                # the 725-image manifest
```

**Run** (after steps (0)-(4) of the Cookbook get you an SSH prompt on a fresh, many-core instance):
```bash
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/task-v3    # PRIVATE: pilot read from $S3_URI/data/
export CALIBRATE=true                                  # fit to pilot instead of the synthetic swept grids
bash run_task_v3_sim.sh 2>&1 | tee run.log
# GT_METHOD=classical for a no-R provisional fit; REPS=N to average more cohorts. USE_ISOTROPIC/decay
# are ignored under CALIBRATE=true (GT comes from the pilot). Edit the num_subjects / design grids in
# run_task_v3_sim.sh's CALIBRATE block to change the swept design. TERMINATE when done.
```
It fails fast if `$S3_URI/data/` is empty, and uploads
`calibration/{calibrate.log, calibrated_params.json, gt_pilot_coords.npy}` + `out/*.csv` + `mds_store/`
to S3. Pull results and view them exactly as above (the calibrated `out/` feeds
`evaluate_simulation.ipynb` unchanged). Keep the whole `$S3_URI` prefix **private** (pilot-derived).

### Task-v4 flavor (screening block, pilot-calibrated)

`run_task_v4_sim.sh` follows the same staging and cookbook as the calibrated v3 flavor above (same
`$S3_URI` layout, same pilot prerequisites) but has **no `CALIBRATE` switch** - it is always
calibrated. Point it at a fresh prefix:

```bash
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/task-v4     # PRIVATE; pilot read from $S3_URI/data/
bash run_task_v4_sim.sh 2>&1 | tee run.log
# optional: TR_LIST=0.24,0.35,0.5  MINREL_LIST=-1,0,0.2,0.4  DF_LIST=5  DISP=0.2  REPS=6
# TERMINATE the instance when done.
```

**What the design constants mean.** `trials_per_subject` is the **main stage only** (14 = 12
distinct + 2 repeats); the screening block's 8 trials are simulated separately and pooled in for
retained subjects, so the effective session is 22 trials / 360 distinct images - matching the
deployed `task_config.json`. Each candidate's images are drawn as one pool and partitioned across
the two stages, so no image appears in both (as `trial_generator.js::partitionIntoStages`
guarantees).

**Reading the output.** Per `out/df<df>_tr<RR>/`:

| File | Question it answers |
|---|---|
| `coverage.csv` | Recruitment cost (`n_candidates_screened`, `screening_pass_rate`) and the **retained** cohort's realised `mean_test_retest` / `mean_subject_noise` - the trade screening makes |
| `embedding_stability.csv` | Distance-vector agreement between two cohorts of N (comparable with all previous runs) |
| `embedding_generalizability.csv` | Procrustes M² between the two cohorts' **spaces** (lower = better) |
| `topk_jaccard.csv`, `item_generalizability.csv` | Item-level recoverability for the future "too similar" study |
| `out/plateau_by_df_tr.csv` | Required-N per (ndim, screening threshold), read off both the Spearman and the M² curve |

Group the analysis by the **achieved** reliability, not the target: `calibration/noise_map.csv`
records the achieved *unscreened* R, while screening raises the retained cohort's R, which is
reported per cell in `coverage.csv`.

> **Scope caveat.** The generative model has no `num_moves` or arrangement-SD component, so only
> the `min_reliability` screening criterion is simulated - not the deployed move-ratio and
> distance-SD fail-rate criteria. These results bound what *reliability-based* screening can buy.


### Two-stage flavor: GT construction, then designed vs random

The current programme runs as **two EC2 stages that hand off through S3**, plus a third step that
runs locally. They are separate scripts, and separate instances, because stage 2 must never rebuild
the ground truth: rebuilding it would silently change it between arms and between runs.

`prepare_machine.sh` provides `stage_pull <name>` / `stage_push <name>` for exactly this. They also
solve the other problem, which is that `prepare_machine.sh` does `rm -rf "$WORKDIR"` for a clean
clone and an instance is terminated the moment its stage ends: a long loop pushes partials as it
goes, so an eight-hour scan that dies at hour seven keeps the six dimensionalities it finished.

#### Stage 1 - `run_gt_construction.sh`

Chooses the GT dimensionality from evidence (split-half agreement, corroborated by leave-k-out CV
over subjects) and fits the final embedding. Runs on the **41 pre-SHINE pilot subjects only**, and
fails fast if that count has changed.

```bash
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/gt-construction   # PRIVATE; pilot read from $S3_URI/data/
bash run_gt_construction.sh 2>&1 | tee run.log
# optional: NDIMS=2,3,4,5,6,7,8,10,12,15,20  N_DRAWS=50  CV_K=5  CV_FOLDS=40  SEED=0
# TERMINATE the instance when done.
```

1100 split-half fits + 440 CV fits, roughly 1.5 h at `N_JOBS=10` on a c7i.4xlarge - but **budget
2x**: a ~20-subject half sits at ~26% coverage and hits `max_iters` far more often than the dense
41-subject aggregate, and a max-iters fit pays the full 1000 iterations. `scan.csv` carries
`status_a`/`status_b` and `niter_a`/`niter_b` precisely so the first pushed partial can be checked
and the run killed early if that rate is high.

Fully resumable: `stage_pull gt` runs first, and any dimensionality already in `scan.csv`/`cv.csv`
is skipped. The half-splits are persisted to `gt/splits.npz`, so a resumed run scores the *same*
draws and the comparison across dimensionalities stays paired.

| Output | What it is |
|---|---|
| `gt/selection.json` | The chosen `n_dims` and the evidence. **Supplies `N_DIMS` to every later script** |
| `gt/gt_pre_shine_d{K}.npy` | The final `(725, K)` coordinates, fitted on all 41 subjects |
| `gt/scan.csv`, `gt/scan_summary.csv` | Split-half scores per (ndim, draw), plus solver status |
| `gt/cv.csv`, `gt/cv_summary.csv` | Held-out Spearman per (ndim, fold) |
| `gt/discard_rates.json` | The split-search diagnostics - **read these** |

> **Read `discard_rates.json` before trusting the scan.** A half is disconnected precisely when it
> holds poorly-covered subjects, so discard-and-redraw is a *biased* filter that over-represents
> well-covered subjects and makes the split-half curve optimistic. Above ~30% discard, or with a
> material `mean_coverage_kept` vs `mean_coverage_discarded` gap, use the leave-k-out curve alone.

#### Stage 2 - `run_design_comparison.sh`

Designed vs random image-to-trial allocation. Pulls stage 1's `gt/` prefix and **fails fast** if
`gt/selection.json` is absent.

```bash
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/design-comparison    # FRESH prefix; PRIVATE
export GT_S3_URI=s3://jon-nir/spam-simulations/gt-construction   # where stage 1 wrote gt/
bash run_design_comparison.sh 2>&1 | tee run.log
# optional: REPS=10 (>=10 enforced)  N_LIST=30,50,75,300  NDIMS=...  MINREL=0.0
# TERMINATE the instance when done.
```

`REPS >= 10` is **enforced, not suggested**: C(6,2)=15 cohort pairs gives visibly wide SEMs, and the
k-selection rule reads those SEMs directly. C(10,2)=45 is the design point.

The D grid defaults to `D_gt + {-3, -1, 0, +2, +5, +10}`, clipped to `[2, 20]` - it must span below,
at and above the selected dimensionality, since a sweep that recovers structure only at exactly the
dimensionality it was generated in has demonstrated nothing. `max(D)` sets the download size (conf
rows are zero-padded to `max_ndim`), so raising it costs twice.

Both arms land in **one store**, with `allocation_mode` (0.0 random / 1.0 designed) as a swept
numeric lever. Every `compute_*` table therefore gains an `allocation_mode` grouping column for
free, and both arms are guaranteed to share a ground truth and a noise draw. Each rep gets a
**fresh** session design; sharing one would leave the designed arm with zero allocation variance
while the random arm carried it, making the two spreads incomparable.

| Output | What it is |
|---|---|
| `out/design_only.csv` | Stage 2a: the arms compared as pure sampling plans (no subjects, no MDS, no R). Runs and is pushed **first** - if the arms do not separate here, 2b has nothing to find |
| `out/coverage.csv`, `out/stability.csv` | Per-cohort coverage and recruitment cost, by arm |
| `out/embedding_generalizability.csv` | Procrustes M² between cohorts' spaces (lower = better) |
| `out/topk_jaccard.csv`, `out/recovery_vs_gt.csv` | Item-level reproducibility, and recovery of the GT's genuinely-closest pairs |
| `out/validity_gradient.csv`, `out/validity.json` | The floor check (below) |
| `out/noise_vs_distance.csv`, `out/noise_curve_shape.csv` | RMSE between a pair's two judgements vs their mean distance, sim and pilot side by side |
| `mds_store/` | ~480 fits, uploaded **without** `confdists.f32` |

> **The validity check is a gate, not a footnote.** If the simulated semantic gradient
> (same-subcategory < same-category < cross-category) does not reproduce, the cohorts are not
> structurally realistic and **the arm comparison should not be trusted**, however clean the arm
> contrast looks. The script prints a warning; do not ignore it.

**Reading `noise_curve_shape.csv`.** Empirically the RMSE between a pair's two judgements rises off
a floor to a late peak (`peak_bin_frac` 0.78) and then drops sharply in the top bin
(`drop_from_peak` 0.37) - clearly-similar and clearly-dissimilar pairs are both judged consistently.
The low-end rise is near-forced (distances cannot go below zero) so any additive-noise model gives
it, and a mismatch *there* is a real alarm. The turnover is the discriminating quantity and needs
the bounded canvas; `drop_gap` reports how far short the model falls on it.

#### Step 3 (local) - `run_cluster_analysis.py`

Pipeline steps g-i. No R, no EC2, a few minutes on one machine. Download the store as above (with
`confs.f32`, without `confdists.f32`), then from the repo root:

```bash
python SpAM_Simulations/run_cluster_analysis.py --store SpAM_Simulations/sim_results/design-comparison/mds_store --out SpAM_Simulations/sim_results/design-comparison/out
```

It writes six frames into `out/`, where `eval_helpers.load_run` picks them up as optionals, and
prints the continuum verdicts plus the density summary.

| File | From | What it holds |
|---|---|---|
| `cluster_agreement.csv` | agglomerative | VI/ARI/AMI, silhouettes, cluster-wise Jaccard per (group, linkage, k) |
| `dendrogram_agreement.csv` | agglomerative | Baker's gamma and cophenetic fidelity, k-free |
| `cluster_sizes.csv` | agglomerative | the discovered size distribution |
| `k_selection.csv` | agglomerative | `k_star_vi`, `k_star_sil` and the continuum verdicts |
| `density_agreement.csv` | HDBSCAN | noise fraction, cross-cohort agreement on isolation, and `vi_restricted`, per `min_cluster_size` |
| `isolated_images.csv` | HDBSCAN | per image, the fraction of cohorts that left it unclustered |

Useful flags: `--ks` and `--linkages` control the agglomerative grid, `--min-cluster-sizes` the
HDBSCAN sweep, and `--density-mcs` picks the single setting `isolated_images.csv` is built at.

**`k_selection.csv` reports two granularities, and they answer different questions.** `k_star_vi` is
selected on VI (one-SE rule) and is the *coarsest granularity that reproduces*: the conservative
deduplication rule, since a coarser cut merges more images and excludes more candidate pairs.
`k_star_sil` is selected on cross-cohort silhouette and is *where the structure actually is*. They
differ because VI measures reproducibility rather than correctness of granularity - on three
well-separated planted blobs VI is exactly 0 at k=2 and at k=3 alike, because both cohorts merge the
same two blobs, and only silhouette distinguishes them (0.93 at the true 3 against 0.76 at 2).

Each k\* is scored on **all three** headline metrics, as `<metric>_at_k_star_vi` and
`<metric>_at_k_star_sil` for `vi_norm`, `sil_cross` and `sil_ratio`. Compare the two columns of a
metric to price the choice: if `vi_norm_at_k_star_sil` is close to `vi_norm_at_k_star_vi`, the finer
granularity costs nothing in reproducibility and should be preferred; if it is much worse, the
parsimonious `k_star_vi` is buying something real.

**The density pass answers a different question, and its VI composes only in a scoped form.**
HDBSCAN is there because agglomerative clustering assigns every one of the 725 images to a cluster,
so a genuinely isolated image is absorbed into whichever group is nearest; HDBSCAN's `-1` noise
label is the missing statement. An image with `frac_cohorts_noise` near 1 is one that no cohort
found reliably confusable with anything, which makes it the safest kind of stimulus.

A labelling with a noise class is not a partition, so `vi_restricted` is computed after dropping the
noise images from every labelling involved. On that shared subset the triangle inequality holds
exactly, so a chained bound is still available in the form "restricted to these n of 725 images".
Two rules when using it: read `n_shared`/`frac_shared` alongside every value, because the subset is
chosen by the clusterings and the score is optimistically biased towards the easy core; and when
adding terms, use `density_clustering.pairwise_restricted_vi`, which puts every pair on one ground
set. The per-pair `mean_vi_restricted` in `density_agreement.csv` is a descriptive average over
pair-specific subsets and is **not** addable. See the README's
[clustering-algorithms section](README.md#clustering-algorithms-why-agglomerative-why-hdbscan-why-not-gmm)
for the full argument, and for why GMM is not an option at all.

> **A continuum is a result.** If `is_flat` (VI varies by less than 0.02 across the whole k grid) or
> `is_arbitrary_slicing` (cross-cohort silhouette at k\* below 0.05) comes back true, the cohorts
> reproducibly agree on a cut of a space that has no separation in it. That means "one image per
> cluster" is the wrong deduplication rule for this data and a distance threshold should be used
> instead. It is a finding, not a failed run.
