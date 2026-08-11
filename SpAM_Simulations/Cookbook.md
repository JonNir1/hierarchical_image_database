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
  - [**Task-v5 flavor: the full two-stage programme (CURRENT)**](#task-v5-flavor-the-full-two-stage-programme-current)

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

Nine shell scripts, under `ec2/`, handle the full-scale sweeps remotely.

> **The v0.1 / v2.3 / v2.4 / v3 / v4 entrypoints are historical.** They drive models that place
> images on an unbounded plane, which the deployed task does not - see
> [the current model is task-v5](README.md#the-current-model-is-task-v5). They are kept because
> their results are cited in [FINDINGS.md](FINDINGS.md) and remain reproducible, not because they
> should be run again. New work uses the two-stage programme at the bottom of this list, which runs
> task-v5.
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
  `N_DIMS` to every later script. See *Task-v5 flavor* below.
- `ec2/run_build_gt_at.sh` - builds one extra ground truth at a given dimensionality
  (`bash run_build_gt_at.sh 8`). Sources `prepare_machine.sh`, so the venv, `PYTHONPATH` and
  `R_LIBS_USER` are right by construction and the pilot data is fetched and scrubbed
  automatically. Needed because stage 2 defaults to a higher-D GT than the scan selects - see
  *Task-v5 flavor* below.
- `ec2/run_design_comparison.sh` - **stage 2 of the current programme.** Designed versus random
  image-to-trial allocation, both arms in one store as a swept `allocation_mode` lever. Pulls
  stage 1's ground truth and never rebuilds it. See *Task-v5 flavor* below.

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


### Task-v5 flavor: the full two-stage programme (CURRENT)

**This is the recipe to use.** Everything above it drives models that place images on an unbounded
plane; see [the current model is task-v5](README.md#the-current-model-is-task-v5).

Three steps: stage 1 and stage 2 run on EC2 (Linux bash), step 3 runs locally (Windows PowerShell).
They hand off through S3 via `stage_pull`/`stage_push`, so no state has to survive an instance.

| | where | what | wall |
|---|---|---|---|
| Stage 1 | EC2 | GT dimensionality + the ground-truth embedding | ~1.5-3 h |
| Stage 2 | EC2 | calibration + the full v5 sweep | ~15 h + generation |
| Step 3 | local | cluster + density analysis | ~10 min |

> **Stage 1 is model-agnostic.** It runs no simulation at all - only weighted MDS on empirical pilot
> distances, which the deployed task already canvas-normalises. If a usable `gt/selection.json`
> already exists from a previous run, skip to stage 2 and point `GT_S3_URI` at it.

#### A. Stage the pilot data (local, PowerShell)

The CSVs and manifest are gitignored, so each EC2 script pulls them from S3 and deletes them from
the box on exit. **Both prefixes need their own copy**: each script reads `$S3_URI/data`, and stage 2
reads its *own* prefix, not stage 1's.

```powershell
cd C:\Users\nirjo\Documents\University\PhD\Projects\hierarchical_image_database
$GT_URI  = "s3://jon-nir/spam-simulations/gt-construction-v5"
$RUN_URI = "s3://jon-nir/spam-simulations/design-comparison-v5"

foreach ($u in @($GT_URI, $RUN_URI)) {
  aws s3 cp data/ "$u/data/" --recursive --exclude "*.pdf" --exclude "*prolific_receipt*"
  aws s3 cp SpAM_Task/stimuli_manifest.json "$u/data/"
}
```

> **PRIVATE prefixes.** These hold human-subjects data. The `--exclude` flags matter: payment
> receipts have historically been uploaded by accident because an earlier recipe's exclusion never
> took effect. Verify with `aws s3 ls "$GT_URI/data/" --recursive | Select-String pdf` - it should
> return nothing.

#### B. Launch and connect

Follow steps **(0)-(4)** of [the allocate/run/verify/terminate
cookbook](#cookbook-allocate---run---verify---terminate) above, with `$INSTANCE_TYPE =
"c7i.4xlarge"` (16 vCPU). Step (3) copies `SpAM_Simulations\ec2\*.sh` to the instance home dir; both
stage scripts `source` `prepare_machine.sh` by relative path, so they must travel together.

> **`git push` does not update the entrypoints.** The `run_*.sh` scripts are **scp'd** to `~`; only
> the Python package is cloned. So after changing any entrypoint you must re-copy it - pushing alone
> leaves the box running an old script against a new package, and the failure is silent: the sweep
> starts normally and produces results for the wrong configuration.
>
> ```powershell
> scp -i $KEY_PATH SpAM_Simulations\ec2\*.sh ubuntu@${IP}:~
> ```
>
> `prepare_machine.sh` now compares the scp'd scripts against the same files at `$GIT_REF` and prints
> a 10-second **STALE ENTRYPOINT(S)** banner if they differ, so this is caught at provisioning rather
> than after a run. A clean box prints `>> entrypoints match main`.

> **Use On-Demand for stage 2, not Spot.** A 15-hour run has a real chance of being reclaimed. If
> you do use Spot, the run is resumable - `run_mds_sweep` skips tasks already in the store and stage
> 2 does `stage_pull mds_store` at the top - so re-launching and re-running the same command
> continues rather than restarts. Stage 1 is equally resumable via `stage_pull gt`.

#### C. Stage 1 (EC2, bash)

Run inside `tmux` so an SSH drop does not kill it.

```bash
tmux new -s gt
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/gt-construction-v5
bash run_gt_construction.sh 2>&1 | tee run.log
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t gt`.

**Three gates before trusting the result.** All three are printed to the log and pushed to S3.

1. **Subject count.** The script hard-fails unless `load_pilot_subjects(variants=("pre",))` resolves
   exactly **41**. A silently different subject set would invalidate every downstream comparison.
2. **Coverage gap, not discard rate** (`gt/discard_rates.json`). Disconnected half-splits are
   redrawn, which *could* bias the scan - but only if the filter is selective. Read
   `coverage_gap_frac` (kept minus discarded binding-half coverage, relative to the kept level);
   above **5%** the split-half curve is genuinely optimistic and leave-k-out should be used alone.
   A high `discard_rate` with a negligible gap means the pool is sparse, not the estimate biased,
   and is expected here: 25 of the 41 pre-SHINE subjects ran v1/v2 at 10 trials each against v3's
   18, so halves heavy on them are disproportionately disconnected without being lower-coverage.
   Measured on this pilot: ~40% discarded, gap +0.4%.
3. **max-iters rate**, in the first pushed `gt/scan.csv` partial. A ~20-subject half sits at ~26%
   coverage and hits the solver's iteration cap far more often than the dense 41-subject aggregate,
   and a max-iters fit pays the full 1000 iterations. If most fits never converged, the scan is
   measuring the stopping rule rather than the data - kill it early.

> **A fresh SSH session has none of the run's environment.** `prepare_machine.sh` activates the
> venv, sets `PYTHONPATH` and `cd`s into the clone inside its *own* shell, so a new session has no
> `python` on PATH (and no system `pip`), cannot import `SpAM_Simulations`, and starts in `$HOME`.
> Reading files needs only the `cd`; running anything needs all of:
>
> ```bash
> cd ~/spam_run/repo && source .venv/bin/activate && export PYTHONPATH="$PWD" \
>   && export R_LIBS_USER="$HOME/R/library"
> ```

`gt/` lives inside the clone, **not** in `$HOME` - `prepare_machine.sh` does `cd "$WORKDIR"` then
`cd repo`, so a fresh shell (or a `tmux` detach/reattach) starts two levels up. Either `cd` there:

```bash
cd ~/spam_run/repo                       # $WORKDIR/repo; override WORKDIR and this moves too
cat gt/selection.json                    # the chosen n_dims, and every rule's choice
cat gt/discard_rates.json                # gate 2
column -s, -t gt/scan_summary.csv        # the split-half curve
column -s, -t gt/cv_summary.csv          # the leave-k-out curve, for gate 4
```

or read them straight from S3, which works from any directory and after the instance is gone:

```bash
for f in selection.json discard_rates.json scan_summary.csv cv_summary.csv; do
  echo "=== $f ==="
  aws s3 cp "$S3_URI/gt/$f" -
done
```

**If split-half and leave-k-out disagree materially on `n_dims`, stop and report** rather than
proceeding. They ask different questions - agreement between halves versus generalisation to unseen
people - so their agreeing is real corroboration and their disagreeing is a finding.

#### C2. Build the higher-D ground truth (EC2, bash)

**The scan's choice is a floor, not an estimate, and stage 2 deliberately does not use it.** The
scan picks the dimensionality at which two 20-subject halves still agree; above D~4 the fit has more
freedom than 17%-coverage halves constrain, so the curve falls from overfitting rather than from the
higher dimensions being empty. On the pilot it chose **3**, while the top-5% closest-pair Jaccard was
still climbing at D=20 and peak global agreement was only ρ=0.233 - about 5% shared rank variance,
i.e. power was binding, not geometry.

A 3-D ground truth is *easier* to recover than the truth, so a planning simulation built on it
understates required-N. Stage 2 therefore defaults to `GT_NDIM=8`. Build that GT in the stage-1
clone, where R and the pilot data already are - it is a single SMACOF fit, a couple of minutes:

A fresh SSH session inherits none of the run's environment - `prepare_machine.sh` sets it up inside
the script's own shell - so all three of these are needed before any in-clone Python will work:

```bash
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/gt-construction-v5
bash run_build_gt_at.sh 8
```

`run_build_gt_at.sh` sources the same `prepare_machine.sh` the sweeps use, so the venv, `PYTHONPATH`,
`R_LIBS_USER` and working directory are all correct **by construction**. Doing this by hand needs
those four plus a re-fetch of the pilot data, which every entrypoint's exit trap deliberately
scrubs - and getting any one wrong costs a round trip. The script's own trap scrubs the data again
on exit, so the box is left clean whether it succeeds or fails.

Several dimensionalities, if you want a GT-dimensionality sensitivity arm later:

```bash
for d in 5 8 12; do bash run_build_gt_at.sh "$d"; done
```

> **A trailing R warning is expected and is not a failure.** After the fit completes, rpy2 prints
> `libraries '/usr/local/lib/R/site-library', '/usr/lib/R/site-library' contain no packages` as R
> shuts down. It only says the *default* library paths are empty, which they are - `smacof` lives in
> `$R_LIBS_USER`, and the fit already used it. Judge success by the `[extra-gt] wrote ...` line and
> the `>> DONE.` banner, not by the absence of R noise.

It writes `gt/gt_pre_shine_d8.npy` and records the build in `gt/extra_gts.json`. It does **not**
touch `selection.json`, which stays as the record of what the evidence chose; the departure from it
lives in stage 2's `GT_NDIM`, where it is documented. Set `GT_NDIM=selected` to follow the scan
instead.

Terminate the instance (step (11)) or keep it for stage 2.

#### D. Stage 2 (EC2, bash)

```bash
tmux new -s sweep
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/design-comparison-v5
export GT_S3_URI=s3://jon-nir/spam-simulations/gt-construction-v5
bash run_design_comparison.sh 2>&1 | tee run.log
```

Optional overrides, all with defaults that encode the decisions already made:

```bash
export REPS=10                       # >=10 enforced; C(10,2)=45 cohort pairs
export N_LIST=30,50,75,500           # 500 is a CEILING PROBE, not a recruitment target
export MINREL_LIST=-1,0,0.1,0.2      # -1 = no-exclusion control; 0.4 omitted (1.9% pass rate)
export SOFTNESS_LIST=3,4,8           # canvas-wall sensitivity arm (4 is calibrated)
export NDIMS=...                     # override the D grid derived from gt/selection.json
```

`perspective_dispersion` is deliberately **not** an env var: it is fitted from the pilot, then swept
±0.15 around the fit, clamped to the fitter's own [0, 1.2] search range.

**What to check in the first push**, roughly 20 minutes in, before committing to the remaining ~15
hours:

```bash
aws s3 cp "$S3_URI/calibration/calibration.json" - | cat   # the fitted constants
aws s3 cp "$S3_URI/out/design_only.csv" - | head -5        # stage 2a, banked first
```

* **`dispersion_swept`** - if it holds only **two** values the fit landed on a clamp boundary (0 or
  1.2). Not fatal, but the sensitivity arm is then one-sided and should be reported as such.
* **`screening_pass_rate`** in `out/coverage.csv` once it appears. Under v4, `min_rel=0.2` passed
  13.1% of candidates, i.e. ~7.6 screened per retained subject. v5 shifts the reliability
  distribution, and the calibration should re-centre it - but if a cell rejects far harder it will
  hit `MAX_RECRUIT_PER_SUBJECT=500` and abort with a message naming the achieved rate.
* **`noise_shape_fit.json`'s `at_shape_boundary`** - if true, the noise *family* rather than the
  data is the binding constraint, and the required-N numbers should not be trusted until the grid
  is widened.

> **The semantic-gradient check is a gate, not a footnote.** If the simulated gradient
> (same-subcategory < same-category < cross-category) does not reproduce, the cohorts are not
> structurally realistic and **the arm comparison should not be trusted**, however clean it looks.
> The script prints a warning near the end.

Then terminate the instance - step (11) of the cookbook above.

**If stage 2 dies after the sweep but before the tables**, do not re-run it. `coverage.csv` and
`stability.csv` are computed and pushed *before* the sweep, so everything still missing derives from
the store alone:

```bash
aws s3 sync "$S3_URI/mds_store/" mds_store/ --only-show-errors
TABLES_ONLY=1 bash run_design_comparison.sh 2>&1 | tee recover.log
```

That skips cohort generation and the sweep entirely - minutes instead of the ~1.4 h a regeneration
costs. `REUSE_CALIBRATION=1` (the default) also skips the ~10 min re-fit when no fitted input changed;
the cache is keyed on the GT contents, the subject sample, the grids and the softness, so a changed
input still forces a re-fit.

#### E. Download (local, PowerShell)

Pull `confs.f32`, never `confdists.f32`. The latter is exactly `pdist(conf)` per row and ~20x
larger; newer runs do not upload it at all, so the exclusion is belt-and-braces.

```powershell
cd C:\Users\nirjo\Documents\University\PhD\Projects\hierarchical_image_database
$RUN  = "design-comparison-v5"
$DEST = "SpAM_Simulations\sim_results\$RUN"
aws s3 sync "s3://jon-nir/spam-simulations/$RUN/out/"         "$DEST\out\"         --only-show-errors
aws s3 sync "s3://jon-nir/spam-simulations/$RUN/calibration/" "$DEST\calibration\" --only-show-errors
aws s3 sync "s3://jon-nir/spam-simulations/$RUN/mds_store/"   "$DEST\mds_store\"   --exclude "confdists.f32" --only-show-errors
Get-ChildItem -Recurse $DEST | Select-Object FullName
```

`sim_results/` is gitignored, so these stay local.

#### F. Cluster analysis (local, PowerShell)

```powershell
.venv\Scripts\python.exe -m SpAM_Simulations.run_cluster_analysis --store SpAM_Simulations\sim_results\design-comparison-v5\mds_store --out SpAM_Simulations\sim_results\design-comparison-v5\out
```

Needs no R. Writes six frames into `out/`, which `eval_helpers.load_run` then picks up as optional
frames for the notebook.

**Budget hours, not minutes.** On a full stage-2 store (1,728 groups x 10 fits x 725 images) this took
~4h50m on 8 cores with everything parallel. It is memory-bandwidth bound, so more cores help less than
you would expect. `--n-jobs 1` for a serial run, `--max-pairs 0` to compare all `C(10,2)` rep pairs
instead of the default 22 (slower, tighter SEM).

| File | From | What it holds |
|---|---|---|
| `cluster_agreement.csv` | agglomerative | VI/ARI/AMI, silhouettes, cluster-wise Jaccard per (group, linkage, k) |
| `dendrogram_agreement.csv` | agglomerative | Baker's gamma and cophenetic fidelity, k-free |
| `cluster_sizes.csv` | agglomerative | the discovered size distribution |
| `k_selection.csv` | agglomerative | `k_star_vi`, `k_star_sil`, and the continuum verdicts |
| `density_agreement.csv` | HDBSCAN | noise fraction, agreement on isolation, `vi_restricted` |
| `isolated_images.csv` | HDBSCAN | per image, the fraction of cohorts that left it unclustered |

`k_selection.csv` carries one row per **configuration x linkage**, not per reporting axis, and
`high_k` / `k_star_*_is_high_k` flag cuts at k >= 150 - under five images per cluster at 725 images,
which is the granularity the pilot supports least.

#### G. Ground-truth diagnostics (local, PowerShell)

Answers whether the GT is a faithful summary of the data it was fitted on. Run it before trusting any
fine-grained result: `frac_of_ceiling` above 1 means the embedding reproduces variance the raw data
cannot reproduce in itself.

```powershell
.venv\Scripts\python.exe -m SpAM_Simulations.gt_diagnostics --gt SpAM_Simulations\sim_results\design-comparison-v5\gt_pre_shine_d8.npy --manifest SpAM_Task\stimuli_manifest.json --out SpAM_Simulations\sim_results\design-comparison-v5\gt_diagnostics
```

#### H. The noise-shape validity check (local, PowerShell)

Needs no store and no R - it reads `calibration.json` and the pilot repeat trials, and runs in
seconds. Stage 2 also runs it, but having it locally means a crash at the tail of a 15-hour EC2 script
costs nothing.

```powershell
.venv\Scripts\python.exe -m SpAM_Simulations.run_validity --run SpAM_Simulations\sim_results\design-comparison-v5 --manifest SpAM_Task\stimuli_manifest.json
```

#### I. Rebuild the store-derived tables (local, only after a grouping fix)

```powershell
.venv\Scripts\python.exe -m SpAM_Simulations.recompute_store_tables --run SpAM_Simulations\sim_results\design-comparison-v5
```

Rebuilds `embedding_stability`, `embedding_generalizability`, `topk_jaccard` and `recovery_vs_gt`
against the current grouping. ~3 hours on 8 cores; `--only <name>` for a subset.

#### J. The report (local, PowerShell)

```powershell
.venv\Scripts\python.exe -m SpAM_Simulations.build_report --run SpAM_Simulations\sim_results\design-comparison-v5
```

Writes a self-contained `report_v5.html` (~5 MB, plotly inlined, opens anywhere). `report_v5.ipynb`
reproduces it cell by cell.

#### Full-run checklist

```
[ ] data staged under BOTH S3 prefixes, no PDFs
[ ] stage 1: 41 pre-SHINE subjects asserted
[ ] stage 1: discard rate < ~30%
[ ] stage 1: max-iters rate acceptable in the first partial
[ ] stage 1: split-half and leave-k-out agree on n_dims
[ ] higher-D GT built and pushed (build_extra_gt --ndim 8), or GT_NDIM=selected chosen deliberately
[ ] stage 2: dispersion_swept has 3 values (2 = clamped, report as one-sided)
[ ] stage 2: no cell hit MAX_RECRUIT_PER_SUBJECT
[ ] stage 2: noise_shape_fit at_shape_boundary is false
[ ] stage 2: semantic gradient monotone on the supported levels
[ ] stage 2: calibration reused or re-fitted deliberately (REUSE_CALIBRATION)
[ ] local: k_selection.csv has one row per configuration x linkage, not tens of thousands
[ ] local: _grouped_successful reported 0 duplicates, or the store was resumed and they were dropped
[ ] BOTH instances terminated
[ ] downloaded with confs.f32, without confdists.f32
```
