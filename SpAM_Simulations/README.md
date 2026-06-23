# SpAM_Simulations

Simulate SpAM experiments (noisy subject distance judgements over a ground-truth embedding)
and reconstruct them with weighted MDS, to evaluate how well MDS recovers the latent space
under different sampling/noise regimes.

## Modules

| Module | Responsibility |
|---|---|
| `experiment.py` | Core simulation: `simulate_experiment` / `simulate_single_subject` (vectorized, condensed form). |
| `design.py` | Per-subject trial allocation (`compute_design_counts`, `build_trial_lists`) for the realistic simulation, ported from `SpAM_Task`'s `buildTrialLists`. |
| `realistic_experiment.py` | Realistic simulation: per-subject image subset + trial design (matches `SpAM_Task`), plus the within-subject SNR heuristic. |
| `simulation.py` | `Simulation` container + ground-truth distances; `make` (random) / `from_embeddings` (real data). |
| `metrics.py` | `coverage`, `spearman_correlation`, `snr_summary`. |
| `helpers.py` | Distance-matrix format conversion (`convert_to_condensed`). |
| `multi_dimensional_scaling.py` | `run_mds` - weighted SMACOF via R's `smacof` (needs R + rpy2). |
| `config.py` | `SimulationConfig`, `RealisticSimulationConfig`, `MDSSweepConfig` - declarative study configuration. |
| `pipeline.py` | Reusable orchestration (generate / coverage / stability / MDS sweep / embedding stability) for both simulation types. |
| `storage.py` | `ResultStore` - compact, streamable, resumable on-disk store for sweep results. |
| `example_pipeline.py` | Minimal runnable end-to-end example. |
| `evaluation.ipynb` | Plotting / analysis notebook for the uniform simulation. |
| `evaluation_realistic.ipynb` | Plotting / analysis notebook for the realistic simulation. |
| `prepare_machine.sh`, `run_uniform_sim.sh`, `run_realistic_sim.sh` | EC2 provisioning + sweep scripts - see "Running on EC2" below. |

## Quick start

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

Three shell scripts handle the full-scale sweeps remotely:
- `prepare_machine.sh` - shared provisioning (system packages, R 4.5 + `smacof`, awscli v2,
  sparse-checkout clone of `SpAM_Simulations/`, Python venv). Sourced, not run directly.
- `run_uniform_sim.sh` - runs the uniform (original) simulation's full-study sweep.
- `run_realistic_sim.sh` - runs the realistic (per-subject trial design) simulation's
  full-study sweep. Heavier than the uniform sweep - size the instance accordingly.

Both entrypoints `source` `prepare_machine.sh` by relative path, so copy all three files
together onto the instance (don't just transfer the one entrypoint you want to run).

### Cookbook: allocate -> run -> verify -> terminate

Known infra for this project: security group `sg-0e1f88c3d550f7154`, key pair `paf-key`
(`.pem` kept outside the repo), IAM instance profile `spam-simulations` (has S3 access).

**(0) One-time setup constants** (PowerShell):
```powershell
$SG_ID         = "sg-0e1f88c3d550f7154"
$KEY_NAME      = "paf-key"
$KEY_PATH      = "C:\Users\nirjo\Documents\projects\__secrets__\paf-key.pem"
$IAM_PROFILE   = "spam-simulations"
$INSTANCE_TYPE = "c7i.4xlarge"     # 16 vCPU; bump to c7i.8xlarge+ for run_realistic_sim.sh
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
$UBUNTU_VERSION = "24.04"     # or "22.04" - matches prepare_machine.sh's own codename fallback (noble)
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
  --instance-market-options '{"MarketType":"spot"}' `   # <- DROP this line for On-Demand
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' `
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spam-ec2}]' `
  --query "Instances[0].InstanceId" --output text

aws ec2 wait instance-running --instance-ids $INSTANCE_ID
$IP = aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text
$IP
```
(100 GB root volume - the realistic sweep's `mds_store/` is larger than the uniform sweep's
~17 GB; shrink back down for uniform-only runs if you want to save a few cents.)

**(3) Transfer the scripts:**
```powershell
scp -i $KEY_PATH `
  SpAM_Simulations\prepare_machine.sh `
  SpAM_Simulations\run_uniform_sim.sh `
  SpAM_Simulations\run_realistic_sim.sh `
  ubuntu@${IP}:~
```

**(4) SSH in:**
```powershell
ssh -i $KEY_PATH ubuntu@$IP
```

**(5) On the instance - set the run-specific constants:**
```bash
export REPO_URL=https://github.com/JonNir1/hierarchical_image_database.git
export GIT_REF=main
export S3_URI=s3://<your-bucket>/spam-mds/run-$(date +%Y%m%d)
# WORKDIR, N_JOBS, R_LIBS_USER all have sane defaults (see each script's header) - override only if needed
```

**(6) Start tmux, run the sweep, log it:**
```bash
tmux new -s spam
bash run_uniform_sim.sh 2>&1 | tee run.log
# or: bash run_realistic_sim.sh 2>&1 | tee run.log
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

**(10) Terminate and confirm:**
```powershell
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID
aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].State.Name" --output text
# expect: terminated
```

## Tests

```
.venv/Scripts/python.exe -m pytest SpAM_Simulations/__tests__ -q
```

R-dependent tests (`test_pipeline_mds.py`) auto-skip if the R bridge can't be imported; the
rest run anywhere. With R configured the full suite is 55 tests.
