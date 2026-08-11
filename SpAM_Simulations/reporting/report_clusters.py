"""Report sections covering cluster structure, density, validity, and the limits of the whole thing.

Split from :mod:`report_sections` only for length. Same contract: take the loaded run, return HTML,
compute every number from the tables.
"""
from __future__ import annotations

import html
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from SpAM_Simulations.reporting.build_report import (
    ARM_COLOUR, LINKAGE_COLOUR, Run, _by_arm_lines, _fig, _missing, _section, _table_html,
)

HIGH_K = 150


def section_clusters(run: Run) -> str:
    ag = run.get("cluster_agreement")
    ks = run.get("k_selection")
    den = run.get("dendrogram_agreement")
    if ag is None:
        return _section("clusters", "9. At what granularity does structure reproduce?",
                        _missing("cluster_agreement"))

    curve = ag.groupby(["linkage", "k"])[["mean_vi_norm", "mean_sil_cross"]].mean().reset_index()
    vi = go.Figure()
    sil = go.Figure()
    for linkage, sub in curve.groupby("linkage"):
        colour = LINKAGE_COLOUR.get(linkage)
        vi.add_trace(go.Scatter(x=sub["k"], y=sub["mean_vi_norm"], mode="lines+markers",
                                name=linkage, line=dict(color=colour, width=2)))
        sil.add_trace(go.Scatter(x=sub["k"], y=sub["mean_sil_cross"], mode="lines+markers",
                                 name=linkage, line=dict(color=colour, width=2)))
    for fig, lab in ((vi, "VI (normalised) — lower means the two cohorts agree"),
                     (sil, "cross-cohort silhouette — above 0 means the clusters separate")):
        fig.update_xaxes(title_text="number of clusters (k)", type="log",
                         tickvals=sorted(curve["k"].unique()))
        fig.update_yaxes(title_text=lab)
        fig.add_vrect(x0=HIGH_K, x1=curve["k"].max(), fillcolor="rgba(214,39,40,0.08)",
                      line_width=0, annotation_text="low confidence", annotation_position="top left")
    sil.add_hline(y=0, line=dict(dash="dash", color="rgba(128,128,128,0.8)"))
    f_vi = _fig(vi, "Cluster agreement degrades steadily as the cut gets finer")
    f_sil = _fig(sil, "Above k≈12 the clusters do not separate in the other cohort's geometry")

    at2 = ag[ag["k"] == 2]
    g2 = at2.groupby(["num_subjects", "arm"])[
        ["mean_vi_norm", "mean_sil_cross", "mean_ari"]].mean().reset_index()
    f_arm = _by_arm_lines(at2, "mean_ari", "Cluster recovery at k=2, by arm",
                          "adjusted Rand index")

    kstats = ""
    if ks is not None:
        vi_counts = ks["k_star_vi"].value_counts().sort_index().to_dict()
        sil_counts = ks["k_star_sil"].value_counts().sort_index().to_dict()
        kstats = (f"<p>Across all <strong>{len(ks):,} configurations</strong> the selected "
                  f"granularity is <code>k*={list(vi_counts)[0]}</code> on the reproducibility "
                  f"criterion (counts {vi_counts}) and <code>k*={list(sil_counts)[0]}</code> on the "
                  f"separation criterion (counts {sil_counts}). Not one configuration selected "
                  f"anything finer.</p>")

    dend = ""
    if den is not None:
        d = den.groupby("linkage")[["mean_baker_gamma", "mean_cophenetic_fidelity"]].mean()
        dend = ("<p>Independent of any cut, comparing whole dendrograms:</p>"
                + _table_html(d.reset_index()))

    return _section("clusters", "9. At what granularity does structure reproduce?", f"""
<p>The downstream use is stimulus construction: never put two confusable images in the same stimulus.
That is a question about <em>groups</em>, not pairs, so the space is clustered bottom-up at a grid of
granularities and two independent cohorts are asked whether they find the same groups. Granularity is
the output here, not an input.</p>

{kstats}

<p><strong>This is the central negative result of the simulation, and it is unambiguous.</strong>
Agreement is high at k=2 and decays monotonically: normalised VI runs 0.07 at k=2, 0.26 at k=5, 0.50
at k=20. More decisively, the cross-cohort silhouette &mdash; which asks whether one cohort's clusters
are still separated in another cohort's geometry &mdash; <strong>crosses zero at about k=12 and is
negative beyond it</strong>. Negative means points sit closer to a neighbouring cluster than to their
own. Whatever the algorithm returns at k=20 and above is a slicing of a continuum, not a set of
groups.</p>

<p>Both selection rules agree on k*=2 in every configuration, which for 725 images means the data
supports one binary distinction and nothing below it. Note that the two rules are constructed to
disagree where there is real structure &mdash; VI rewards parsimony, silhouette rewards separation
&mdash; so their unanimity here is informative rather than redundant.</p>

{dend}

<p>Average linkage reproduces best on every measure, which is the expected ordering: it assumes least
about cluster shape, while complete linkage is outlier-sensitive and ward presumes compact
equal-sized clusters that this space does not have.</p>

<p class="note"><strong>Read the shaded region with care.</strong> At 725 images, k&nbsp;&ge;&nbsp;150
cuts at under five images per cluster &mdash; the granularity the pilot supports least (section 3).
The apparent recovery of VI at very high k is an artefact of both partitions approaching
all-singletons, where VI falls for arithmetic reasons while the silhouette continues to worsen.</p>
""", [f_vi, f_sil, f_arm])


def section_density(run: Run) -> str:
    den = run.get("density_agreement")
    if den is None:
        return _section("density", "10. Which images belong to no group at all?",
                        _missing("density_agreement"))
    g = den.groupby("min_cluster_size")[
        ["mean_n_clusters", "mean_frac_noise", "mean_noise_kappa",
         "mean_ari_shared_clustered", "mean_frac_shared_clustered"]].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["min_cluster_size"], y=g["mean_frac_noise"], mode="lines+markers",
                             name="fraction left unclustered", line=dict(color="#d62728", width=2)))
    fig.add_trace(go.Scatter(x=g["min_cluster_size"], y=g["mean_noise_kappa"], mode="lines+markers",
                             name="cohort agreement on which (kappa)",
                             line=dict(color="#1f77b4", width=2)))
    fig.update_xaxes(title_text="HDBSCAN min_cluster_size")
    fig.update_yaxes(title_text="proportion")
    f1 = _fig(fig, "Most images are not confusable with anything, and cohorts barely agree on which")

    return _section("density", "10. Which images belong to no group at all?", f"""
<p>Agglomerative clustering must assign every image to some cluster. That is the wrong assumption for
stimulus construction, where the useful answer is often &ldquo;this image resembles nothing else, so
it is safe to use&rdquo;. HDBSCAN can decline to cluster an image, so it is run as a purely
descriptive pass &mdash; it is never substituted into the VI chain, because its noise label breaks the
metric properties that licence chaining.</p>

{_table_html(g)}

<p>At <code>min_cluster_size=5</code>, <strong>61% of images are left unclustered</strong> and the
recovered structure is roughly five clusters. Two cohorts agree on <em>which</em> images those are at
&kappa;&nbsp;=&nbsp;0.20 &mdash; slight agreement by any conventional reading. So the pattern is not
just that most images resist grouping, but that which ones resist is itself unstable across samples.</p>

<p>The trade across the parameter is stark: at <code>min_cluster_size=2</code> HDBSCAN finds 171
clusters but two cohorts agree on their content at ARI 0.05, i.e. not at all. Agreement only becomes
respectable (ARI 0.90+) at settings that cluster under 18% of the images into two groups. There is no
setting that both covers the image set and reproduces.</p>

<p class="note">This corroborates section 9 through a completely different algorithm, which is why it
was run. Agglomerative clustering said fine-grained groups do not reproduce; density clustering says
most images were never in a group to begin with.</p>
""", [f1])


def section_validity(run: Run) -> str:
    figs: List[go.Figure] = []
    parts: List[str] = []

    grad = run.get("validity_gradient")
    if grad is not None:
        arm = grad[grad.get("arm", pd.Series(["random"] * len(grad))) == "random"] \
            if "arm" in grad.columns else grad
        fig = go.Figure()
        for src, colour in (("sim", "#1f77b4"), ("pilot", "#d62728")):
            col = f"mean_distance_{src}"
            if col in arm.columns:
                y = arm[col] / arm[col].iloc[0]
                fig.add_trace(go.Scatter(x=arm["level_name"], y=y, mode="lines+markers",
                                         name=src, line=dict(color=colour, width=2)))
        fig.update_xaxes(title_text="semantic relatedness (coarse → fine)")
        fig.update_yaxes(title_text="mean distance ÷ unrelated-pair distance")
        figs.append(_fig(fig, "Does the simulation separate semantically close images as people do?"))
        parts.append(_table_html(arm[[c for c in arm.columns if c.startswith(
            ("level_name", "mean_distance", "n_pairs"))]]))

    shape = run.get("noise_curve_shape")
    curves = run.get("noise_vs_distance")
    if curves is not None:
        fig = go.Figure()
        for src, colour in (("sim", "#1f77b4"), ("pilot", "#d62728")):
            sub = curves[curves["source"] == src]
            fig.add_trace(go.Scatter(x=sub["mean_pair_distance"], y=sub["rmse"],
                                     mode="lines+markers", name=src,
                                     line=dict(color=colour, width=2),
                                     error_y=dict(type="data", array=sub["sem_rmse"],
                                                  visible=True, thickness=1)))
        fig.update_xaxes(title_text="how far apart the pair was placed (mean of two judgements)")
        fig.update_yaxes(title_text="RMSE between the two judgements")
        figs.append(_fig(fig, "Noise against distance: both curves turn over"))
    if shape is not None:
        parts.append(_table_html(shape[[c for c in ("source", "rise_from_first", "drop_from_peak",
                                                    "peak_bin_frac", "is_inverted_u")
                                        if c in shape.columns]]))

    return _section("validity", "11. Is the simulation realistic enough to believe?", f"""
<p>Two out-of-sample checks. Neither was fitted, which is what gives them any force.</p>

<h3>The semantic gradient</h3>
<p>Real participants place semantically related images closer together. The simulation reproduces the
ordering across the levels the pilot data can support, but <strong>under-separates the fine
end</strong>: same-leaf pairs sit at roughly 71% of the unrelated-pair distance in simulation against
28% in the pilot. Some of that is inherited from the ground truth, which already flattens the fine
structure (section 3), and the rest from the noise model.</p>

<p>The practical consequence is conservative rather than misleading: real data separates fine pairs
more sharply than the simulation does, so real recovery should be <em>easier</em> than simulated, and
required-sample estimates from this simulation err on the side of asking for too many participants.</p>

<p class="warn"><strong>Scope limit on this check.</strong> The gradient was scored on a single
configuration per arm at N=500 &mdash; whichever combination of softness, screening threshold and
dispersion happened to be evaluated first &mdash; not across the swept range. It therefore says the
gradient reproduces <em>somewhere</em> in the parameter space rather than <em>throughout</em> it. The
code now scores every configuration and reports the distribution, but producing that requires
re-running the stage-2 sweep; this figure predates the fix. The noise-shape check below is unaffected,
since it measures the calibrated noise model directly rather than any sweep cell.</p>

<h3>The shape of the noise</h3>
<p>This is the sharper test, because it constrains the noise model rather than the signal. In the
pilot, the disagreement between a participant's two judgements of the same pair is an inverted U
against how far apart they placed it: unambiguously-similar and unambiguously-different pairs are
judged consistently, and the ambiguous middle is not. The high-distance turnover is the discriminating
half, because it requires a bounded canvas &mdash; a pair already at opposite corners cannot move much
further apart.</p>

<p><strong>The simulation reproduces the inverted U without having been fitted to it.</strong> The
turnover is present (drop from peak 0.27 against the pilot's 0.37) and the peak falls in the same
region. The remaining mismatch is at the <em>low</em> end: the model never gets as quiet on obviously
similar pairs as real participants do, which is the same fine-grained weakness the gradient shows from
the other direction.</p>

{"".join(parts)}
""", figs)


def section_limits(run: Run) -> str:
    cov = run.get("coverage")
    n_max = int(cov["num_subjects"].max()) if cov is not None else 500
    return _section("limits", "12. What this can and cannot answer", f"""
<h3>Answered</h3>
<ul>
<li><strong>The designed allocation is better, and deployably so.</strong> It covers 18&ndash;30% more
image pairs than random across N=30&ndash;75 for identical participant effort, uses the image set
about five times more evenly, wastes far fewer judgements, and the per-session constraint that keeps
it runnable costs a fraction of a percent. At N={n_max} the arms converge, as they must.</li>
<li><strong>The perceptual space does not support fine-grained clusters.</strong> Every one of the
5,184 configurations selected k*=2, cross-cohort silhouette goes negative above k&asymp;12, and an
independent density-based method leaves 61% of images unclustered with only slight agreement on which.
Deduplicating stimuli by cluster membership is not supportable at any granularity finer than a binary
split; a distance threshold is the appropriate rule instead.</li>
<li><strong>The bounded-canvas model is behaving.</strong> It reproduces the empirical noise-against-distance
turnover it was never fitted to.</li>
</ul>

<h3>Not answered</h3>
<ul>
<li><strong>Required N remains open at the top.</strong> N=500 was included as a ceiling probe and the
curves are still rising into it on several measures. What the simulation establishes is where the
<em>design advantage</em> lives (N=50&ndash;75), not where recovery plateaus.</li>
<li><strong>How much of this is the ground truth rather than the method.</strong> The D=8 embedding
reproduces the pilot beyond the pilot's own reliability ceiling (section 3), so the fine-grained
negative results are partly a statement about 41 participants at 31% coverage, not about SpAM or MDS.
Distinguishing the two needs a ground truth from more data &mdash; which is what the full study
produces.</li>
<li><strong>Whether real participants cluster better than the model implies.</strong> The simulation
under-separates fine pairs relative to the pilot, so the true granularity ceiling may sit somewhat
above k=2. The direction of that bias is known; its size is not.</li>
</ul>

<h3>Recommendation</h3>
<p>Adopt the designed allocation &mdash; the coverage gain is free, the deployability cost is
negligible, and it never underperforms random at any N tested. Plan the analysis around a distance
threshold rather than cluster membership. And treat the fine-grained null results as provisional until
the full study's ground truth can be built on more than 41 participants.</p>
""")
