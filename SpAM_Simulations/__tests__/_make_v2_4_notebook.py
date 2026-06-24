"""One-off generator: build evaluation_task_v2_4.ipynb from evaluation_task_v2_3.ipynb.

Transforms the v2.3 notebook for the task-v2.4 simulation: swaps the config class/grid and
generator call, re-facets the coverage/stability/MDS panels by `frac_trials_repeated` (the new
whole-trial-repeat lever) instead of `frac_images_repeated`, extends the stability `PARAM_FIELDS`
to carry both levers, and replaces the SNR panel with a test-retest reliability panel. Kept under
__tests__/ as a reproducible record of how the notebook was produced (not a pytest module).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "evaluation_task_v2_3.ipynb"
DST = HERE / "evaluation_task_v2_4.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


INTRO = """# Task-v2.4 SpAM Simulation & MDS Evaluation Pipeline

This notebook mirrors `evaluation_task_v2_3.ipynb`, but simulates subjects under the
**task-v2.4** design: the task-v2.3 per-subject trial allocation **plus the
`frac_trials_repeated` whole-trial-repeat lever**. Concretely, on top of everything v2.3 does,
each subject now has `n_repeats = round(frac_trials_repeated * trials_per_subject)` of their
trials shown again verbatim (same `k`-image set). Each repeat **re-draws** its noisy distances
(a fresh, independent arrangement), so the original/repeat pair gives a within-subject
**test-retest reliability**:

$$\\mathrm{rel}_s = \\mathrm{mean}_{\\text{repeated trials}} \\; \\rho\\big(d^{\\text{orig}},\\, d^{\\text{repeat}}\\big)$$

the mean Spearman correlation between the original and repeat pairwise-distance vectors of the
subject's repeated trials (NaN for subjects with no repeats).

**Lever competition.** A repeat may only duplicate a *singles-only* trial (one with no
`frac_images_repeated`-doubled image), so no image exceeds 2 occurrences across both mechanisms.
At `images_per_trial = 20` even a modest `frac_images_repeated` saturates every trial with
doubled images and leaves none to repeat (`select_repeat_trials` raises). This notebook's grid
therefore fixes `frac_images_repeated = 0.0` and sweeps `frac_trials_repeated`, matching the
deployed `task_config.json`. The doubled-image **SNR** heuristic (which needs
`frac_images_repeated > 0`) is characterised in `evaluation_task_v2_3.ipynb` instead.

Everything downstream of trial generation - ground-truth generation, the MDS sweep, and the
coverage/stability metrics - is reused unchanged from `SpAM_Simulations/pipeline.py`.
"""

CONFIG = """# --- Study configuration ----------------------------------------------------------------
# Small configuration for a quick end-to-end run (uncomment to validate the notebook):
# sim_config = TaskV2_4SimulationConfig(
#     n_images=300,
#     n_dims=6,
#     num_subjects=[20, 40, 75],
#     trials_per_subject=[12],
#     images_per_trial=[20],
#     subjects_noise_scale=[0.3, 0.6],
#     subjects_noise_df=[3],
#     frac_images_repeated=[0.0],
#     frac_trials_repeated=[0.0, 0.1, 0.25],
#     reps=3,
#     seed=42,
# )

# Full study configuration (uncomment for the real run - this is much heavier)
# mirrors the deployed task: frac_images_repeated fixed at 0.0, sweep frac_trials_repeated.
sim_config = TaskV2_4SimulationConfig(
    n_images=725,
    n_dims=10,
    num_subjects=[30, 50, 75, 250],
    trials_per_subject=[10, 15, 20],
    images_per_trial=[20],
    subjects_noise_scale=[0.5, 0.8],
    subjects_noise_df=[1],
    frac_images_repeated=[0.0],
    frac_trials_repeated=[0.0, 0.1, 0.2, 0.3],
    reps=5,
    seed=42,
)

sim = pipeline.generate_task_v2_4_simulation(sim_config, verbose=True)
"""

STABILITY = '''PARAM_FIELDS = [
    "num_subjects", "trials_per_subject", "images_per_trial",
    "subjects_noise_scale", "subjects_noise_df", "frac_images_repeated", "frac_trials_repeated",
]

correlations = (
    pipeline.compute_stability_table(sim)
    .dropna(subset=["spearman"])
    .groupby(PARAM_FIELDS)
    .agg(count=("spearman", "size"), r_mean=("spearman", "mean"), r_sem=("spearman", "sem"))
)
correlations.index = correlations.index.set_names(
    ["n_subjects", "trials_per_subject", "images_per_trial",
     "subjects_noise_scale", "subjects_noise_df", "frac_images_repeated", "frac_trials_repeated"]
)
'''

RETEST_MD = """## Subject Test-Retest Reliability
Each subject who receives whole-trial repeats (`frac_trials_repeated > 0`) yields a test-retest
reliability: the mean Spearman correlation between the original and repeat presentations of
their repeated trials - exactly the kind of internal-consistency signal we could also compute
from real (ground-truth-free) data. It is NaN for subjects with no repeated trials (so it is
undefined for the entire `frac_trials_repeated = 0` slice). Two checks below: (1) its
distribution across subjects, and (2) whether it degrades with the configured noise lever - if
it didn't track noise, it would be useless as a real-data QC proxy.
"""

RETEST_HIST = '''# Per-subject test-retest reliability for one representative configuration: most repeated
# trials (so a defined reliability actually exists), most subjects, heaviest-tailed and noisiest.
focus_params = max(
    sim._results,
    key=lambda p: (p.frac_trials_repeated, p.num_subjects, -p.subjects_noise_df, p.subjects_noise_scale),
)
all_rel = np.concatenate([res.subject_test_retest for res in sim._results[focus_params]])
finite_rel = all_rel[np.isfinite(all_rel)]

rel_hist_fig = go.Figure(go.Histogram(x=finite_rel, nbinsx=30))
rel_hist_fig.update_layout(
    width=700, height=400,
    title=dict(text=f"Subject Test-Retest Reliability Distribution<br><sup>{focus_params}</sup>"),
    xaxis=dict(title=dict(text="mean Spearman r (original vs. repeat trial)")),
    yaxis=dict(title=dict(text="Count")),
    template="plotly_white",
)
rel_hist_fig.show()
print(f"{np.mean(np.isnan(all_rel)):.1%} of subjects had no repeated trials (reliability undefined)")
'''

RETEST_VS_NOISE = '''# How the test-retest reliability tracks the configured subject-noise lever, per
# frac_trials_repeated (the frac_trials_repeated == 0 slice is all-NaN and drops out).
rel_vs_noise = (
    coverage[np.isfinite(coverage["mean_test_retest"])]
    .groupby(["subjects_noise_scale", "frac_trials_repeated"])
    .agg(mean_rel=("mean_test_retest", "mean"), sem_rel=("mean_test_retest", "sem"))
    .reset_index()
)

rel_fig = go.Figure()
for frac in sorted(rel_vs_noise["frac_trials_repeated"].unique()):
    df = rel_vs_noise[rel_vs_noise["frac_trials_repeated"] == frac]
    rel_fig.add_trace(go.Scatter(
        x=df["subjects_noise_scale"], y=df["mean_rel"],
        error_y=dict(type="data", array=df["sem_rel"].fillna(0), visible=True),
        name=f"frac_trials_repeated = {frac:.3g}",
        mode="lines+markers",
    ))
del frac, df
rel_fig.update_layout(
    width=700, height=400,
    title=dict(text="Mean Test-Retest Reliability vs. Configured Subject Noise Scale"),
    xaxis=dict(title=dict(text="subjects_noise_scale")),
    yaxis=dict(title=dict(text="mean(test-retest Spearman r)")),
    template="plotly_white",
    legend=dict(title=dict(text="Trial-Repetition Fraction")),
)
rel_fig.show()
'''


def facet_swap(text):
    return (text
            .replace("frac_images_repeated", "frac_trials_repeated")
            .replace("Image-Repetition", "Trial-Repetition"))


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = md(INTRO)
    cells[1]["source"] = [s.replace("TaskV2_3SimulationConfig", "TaskV2_4SimulationConfig")
                          for s in cells[1]["source"]]
    cells[3] = code(CONFIG)
    # Re-facet coverage panels by the new lever.
    for i in (5, 6):
        cells[i]["source"] = facet_swap("".join(cells[i]["source"])).splitlines(keepends=True)
    cells[8] = code(STABILITY)
    cells[9]["source"] = facet_swap("".join(cells[9]["source"])).splitlines(keepends=True)
    # Replace the SNR panel (md + 2 code cells) with the test-retest panel.
    cells[10] = md(RETEST_MD)
    cells[11] = code(RETEST_HIST)
    cells[12] = code(RETEST_VS_NOISE)
    # MDS scree + embedding-stability panels: re-facet by the new lever.
    for i in (17, 19):
        cells[i]["source"] = facet_swap("".join(cells[i]["source"])).splitlines(keepends=True)
    # Point the MDS store at a v2.4-run placeholder.
    cells[14]["source"] = [s.replace('"run-20260623"', '"run-task-v2.4"') for s in cells[14]["source"]]

    # Clear any executed state.
    for c in cells:
        if c["cell_type"] == "code":
            c["execution_count"] = None
            c["outputs"] = []

    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {DST} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
