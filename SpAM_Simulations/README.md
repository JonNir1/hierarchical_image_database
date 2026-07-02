# SpAM_Simulations

Simulate SpAM experiments (noisy subject distance judgements over a ground-truth embedding)
and reconstruct them with weighted MDS, to evaluate how well MDS recovers the latent space
under different sampling/noise regimes.

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
emb   = pipeline.compute_embedding_stability(store)
```

Or run the bundled example: `python -m SpAM_Simulations.example_pipeline` (from the repo root).

### Performance / storage notes
- Generation is vectorized and runs in condensed form (~9-10x faster than the original loop,
  ~half the memory). The serial path is bit-exact and fully reproducible from `seed`.
- `run_mds_sweep(parallel=True, n_jobs=...)` distributes the independent MDS runs across
  processes (joblib/loky), streaming results to disk so peak memory stays bounded.
- `ResultStore` keeps a human-readable `meta.csv` plus a flat float32 `confdists.f32`
  (memory-mapped on read), replacing the old multi-GB append-only pickle.

For real pilot data instead of synthetic, see *Calibrating to pilot data* below. For
large-scale sweeps, see *Running on EC2* below.

## Modules

| Module | Responsibility |
|---|---|
| `experiment.py` | Core simulation: `simulate_experiment` / `simulate_single_subject` (vectorized, condensed form). |
| `design.py` | Per-subject trial allocation (`compute_design_counts`, `build_trial_lists`) for the task-v2.3 simulation, plus `distinct_trial_count`/`select_repeat_trials` for task-v2.4's whole-trial repeats, ported from `SpAM_Task`'s `buildTrialLists`/`insertTrialRepeats`. |
| `task_v2_3_experiment.py` | Task-v2.3 simulation: per-subject image subset + trial design (matches `SpAM_Task`), plus the within-subject SNR heuristic. |
| `task_v2_4_experiment.py` | Task-v2.4 simulation: task-v2.3 design **plus** `frac_trials_repeated` whole-trial repeats (each repeat re-draws its noisy distances), yielding a per-subject test-retest reliability. Bit-exact to task-v2.3 when `frac_trials_repeated=0`. |
| `task_v3_experiment.py` | Task-v3 simulation: a **generative coordinate-space** model replacing additive-distance-noise. Per subject: a perspective weighting of the ground-truth PCs (`perspective_dispersion`) + item-level coordinate noise, projected onto a **local per-trial 2-D arrangement** (the SpAM canvas bottleneck). Drops `frac_images_repeated` (task v3.0); keeps `frac_trials_repeated` test-retest. |
| `simulation.py` | `Simulation` container + ground-truth distances; `make` (random) / `from_embeddings` (real data) / `build_ground_truth_embeddings` (synthetic with a chosen eigenvalue spectrum for task-v3). |
| `metrics.py` | `coverage`, `spearman_correlation`, `snr_summary`, `test_retest_summary`, `effective_rank` (classical-MDS rank of an aggregate - checks the task-v3 2-D slices span >2 dims). |
| `helpers.py` | Distance-matrix format conversion (`convert_to_condensed`). |
| `multi_dimensional_scaling.py` | `run_mds` - weighted SMACOF via R's `smacof` (needs R + rpy2). |
| `config.py` | `SimulationConfig`, `TaskV2_3SimulationConfig`, `TaskV2_4SimulationConfig`, `TaskV3SimulationConfig`, `MDSSweepConfig` - declarative study configuration. |
| `pipeline.py` | Reusable orchestration (generate / coverage / stability / MDS sweep / embedding stability) for all simulation types. |
| `storage.py` | `ResultStore` - compact, streamable, resumable on-disk store for sweep results. |
| `pilot.py` | **Read-only** pilot ingestion + calibration: load `data/pilot/` (via `analysis/pilot/parser.py`), the test-retest / between-subject-agreement observables, the pooled-pilot GT embedding, and `calibrate_params_from_pilot` (the end-to-end fit of `subjects_noise_scale` + `perspective_dispersion`). See "Calibrating to pilot data" below. |
| `example_pipeline.py` | Minimal runnable end-to-end example. |
| `eval_helpers.py` | Read-only loading/plotting helpers for `evaluate_simulation.ipynb` - no simulation, no MDS, no R. |
| `evaluation.ipynb` | Plotting / analysis notebook for the task-v0.1 simulation. |
| `evaluation_task_v2_3.ipynb` | Plotting / analysis notebook for the task-v2.3 simulation. |
| `evaluation_task_v2_4.ipynb` | Plotting / analysis notebook for the task-v2.4 simulation (adds the test-retest reliability panel). |
| `evaluate_simulation.ipynb` | Read-only overview/drill-down figures for an already-completed **task-v3** run (incl. the plateau-N required-subjects readout), via `eval_helpers.py`. v3-only; older runs use the `evaluation*.ipynb` notebooks above. |
| `ec2/prepare_machine.sh`, `ec2/run_task_v0_1_sim.sh`, `ec2/run_task_v2_3_sim.sh`, `ec2/run_task_v2_4_sim.sh`, `ec2/run_task_v3_sim.sh` | EC2 provisioning + sweep scripts - see "Running on EC2" below. `run_task_v3_sim.sh` also has a `CALIBRATE=true` flavor that fits the simulation to the pilot before sweeping (see "Running on EC2 / Calibrated flavor"). |
| `sim_results/<run-name>/` | Local copy of a completed run's small files (`out/*.csv`, `mds_store/meta.csv`) downloaded from S3, e.g. `sim_results/task-v2.3/` - gitignored, consumed by `eval_helpers.py`/`evaluate_simulation.ipynb`. |

## Calibrating to pilot data

By default the task-v3 simulation's internals are *guessed*: `subjects_noise_scale` is scaled to the
synthetic GT signal, but `perspective_dispersion` and the GT-geometry knobs (`decay`/`n_clusters`)
have no anchor - so the absolute required-N is only as meaningful as those guesses. `pilot.py` anchors
all of them to real pilot data, turning the estimate from "as a function of guessed internals" into a
calibrated number.

**The idea (three anchors).** The pilot CSVs (`data/pilot/`, one per session) give three observables:
1. **Ground-truth geometry** - pool *all* completed pilot subjects into one aggregate RDM and run
   weighted SMACOF -> the recovered embedding *is* the GT (it inherits the real eigenvalue spectrum
   and cluster structure, so `decay`/`n_clusters` become moot; `n_dims` is read from the spectrum).
2. **`subjects_noise_scale`** is pinned by within-subject **test-retest** (the v3.0 whole-trial
   repeats). Test-retest is *perspective-invariant* - a subject's perspective is identical across
   their original and repeat trials, only the noise re-draws - so it isolates measurement noise.
3. **`perspective_dispersion`** is then pinned by **between-subject agreement** - correlating subjects'
   RDMs over their jointly-judged pairs - with noise held fixed. This is pooled over *all* pilot
   subjects (not just v3): agreement is a population property, so more subjects = more dyads = a tighter
   anchor. (Only the test-retest step is v3-specific, since it needs the whole-trial repeats.)

Identifiability is therefore sequential and clean: test-retest -> noise, then agreement -> dispersion.

**Running it.** Calibration is a short (~1-5 min) prelude to the sweep, not a separate run: use the
calibrated flavor of the EC2 sweep - `run_task_v3_sim.sh` with `CALIBRATE=true` (see
[*Running on EC2*](#running-on-ec2)) - which fits the parameters + pilot GT and then runs the calibrated
convergence sweep in one go. Programmatically it is a single call:
```python
from SpAM_Simulations.pilot import calibrate_params_from_pilot
coords, fit, info = calibrate_params_from_pilot(
    "data/pilot", "SpAM_Task/stimuli_manifest.json",
    gt_method="smacof",           # or "classical" for a no-R provisional GT (unreliable; smoke-test only)
    save_gt="gt.npy", save_params="params.json",   # optional: persist the artifacts the sweep consumes
)   # -> fitted {subjects_noise_scale, perspective_dispersion, n_dims, ...}
```
The read-only building blocks it composes (`load_pilot_subjects`, `within_subject_test_retest`,
`between_subject_agreement`, `build_gt_from_pilot`) also live in `pilot.py`.

> **Data policy.** `data/pilot/` is human-subjects data: gitignored, **never committed or pushed**.
> Pilot-derived artifacts (the aggregate RDM, the GT `coords`, fitted params) are equally local - keep
> them under the private `$S3_URI` prefix, never in the repo.

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

Five shell scripts, under `ec2/`, handle the full-scale sweeps remotely:
- `ec2/prepare_machine.sh` - shared provisioning (system packages, R 4.5 + `smacof`, awscli v2,
  sparse-checkout clone of `SpAM_Simulations/`, Python venv). Sourced, not run directly.
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

# Pull only the small files the read-only notebook needs: out/*.csv + mds_store/meta.csv.
aws s3 sync "$S3_URI/out/"       "$DEST\out\"       --only-show-errors
aws s3 sync "$S3_URI/mds_store/" "$DEST\mds_store\" --exclude "confdists.f32" --only-show-errors
# `confdists.f32` (the reconstructed embeddings) is multi-GB and is NEVER read by
# eval_helpers.py - drop the `--exclude` only if you need it for ad-hoc analysis.

Get-ChildItem -Recurse $DEST | Select-Object FullName    # expect out/{coverage,stability,embedding_stability}.csv + mds_store/meta.csv
```
`sim_results/` is gitignored, so these stay local. Then open
`SpAM_Simulations/evaluate_simulation.ipynb` and set `RUN_RESULTS_DIR = "sim_results/task-v2.4"`.

### Calibrated flavor (fit to pilot first)

`run_task_v3_sim.sh` with `CALIBRATE=true` runs a short (~1-5 min) prelude that fits the simulation to
the real pilot, then runs the **same** convergence sweep with the pilot GT + fitted
`subjects_noise_scale`/`perspective_dispersion` (see *Calibrating to pilot data* above for the method).
Everything for the study lives under one **private** `S3_URI` prefix, the pilot path mirroring the local
`data/pilot/`:
```
$S3_URI/data/pilot/   <- INPUT  you stage once: session + demographics CSVs + stimuli_manifest.json
$S3_URI/calibration/  <- OUTPUT calibrate.log, calibrated_params.json, gt_pilot_coords.npy
$S3_URI/out/  $S3_URI/mds_store/   <- OUTPUT of the convergence sweep
```

**Stage the pilot once** (the CSVs + manifest are gitignored, so never in the repo clone - the
entrypoint pulls them from S3 and deletes them from the box at exit):
```powershell
$S3_URI = "s3://jon-nir/spam-simulations/task-v3"
aws s3 cp data/pilot/                     "$S3_URI/data/pilot/" --recursive   # session + demographics CSVs
aws s3 cp SpAM_Task/stimuli_manifest.json "$S3_URI/data/pilot/"               # the 725-image manifest
```

**Run** (after steps (0)-(4) of the Cookbook get you an SSH prompt on a fresh, many-core instance):
```bash
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://jon-nir/spam-simulations/task-v3    # PRIVATE: pilot read from $S3_URI/data/pilot/
export CALIBRATE=true                                  # fit to pilot instead of the synthetic swept grids
bash run_task_v3_sim.sh 2>&1 | tee run.log
# GT_METHOD=classical for a no-R provisional fit; REPS=N to average more cohorts. USE_ISOTROPIC/decay
# are ignored under CALIBRATE=true (GT comes from the pilot). Edit the num_subjects / design grids in
# run_task_v3_sim.sh's CALIBRATE block to change the swept design. TERMINATE when done.
```
It fails fast if `$S3_URI/data/pilot/` is empty, and uploads
`calibration/{calibrate.log, calibrated_params.json, gt_pilot_coords.npy}` + `out/*.csv` + `mds_store/`
to S3. Pull results and view them exactly as above (the calibrated `out/` feeds
`evaluate_simulation.ipynb` unchanged). Keep the whole `$S3_URI` prefix **private** (pilot-derived).

## Tests

```
.venv/Scripts/python.exe -m pytest SpAM_Simulations/__tests__ -q
```

R-dependent tests (`test_pipeline_mds.py`) auto-skip if the R bridge can't be imported; the
rest run anywhere. With R configured the full suite is 130 tests.

## Tags

Named snapshots of the repo at notable `SpAM_Simulations` milestones:

| Tag | Commit | Date | Marks |
|---|---|---|---|
| `spam-sim-pre-refactor` | `73a9b14` | 2026-06-18 | Last commit before the bit-exact vectorization + reusable pipeline/storage refactor (condensed-form simulation, ~9x speed-up; `ResultStore`; parallel MDS sweep; EC2 provisioning scripts). |
| `sim-v2.3` | `457fa5f` | 2026-06-24 | `evaluate_simulation.ipynb` display/correctness polish (rounded lever values, float-safe `filter_to_config`, tiered subplot grid via `_grid_dims`) plus the `ec2/`/`sim_results/` directory reorg. |
