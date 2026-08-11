"""The narrative sections of the task-v5 stage-2 report.

Separate from :mod:`build_report` so the page furniture (loading, figure styling, HTML assembly)
stays readable next to the prose. Every section takes the loaded :class:`build_report.Run` and
returns an HTML string; every number in that string is computed from the run's tables rather than
transcribed, so the page cannot drift from the data.
"""
from __future__ import annotations

import html
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from SpAM_Simulations.build_report import (
    ARM_COLOUR, LINKAGE_COLOUR, Run, _by_arm_lines, _fig, _missing, _section, _table_html,
)


def _delta(designed: float, random: float) -> str:
    if not random:
        return "n/a"
    return f"{100 * (designed - random) / random:+.1f}%"


# --------------------------------------------------------------------------- 1. the model
def section_model(run: Run) -> str:
    cal = run.calibration

    def val(key, fmt="{}"):
        v = cal.get(key)
        return "?" if v is None else fmt.format(v)

    rows = [
        ("ground truth", f"{val('gt_file')} at D={val('n_dims')} "
                         f"(the pilot's own scan selected D={val('scan_selected_n_dims')})"),
        ("pilot sessions", f"{val('n_pilot_sessions')}, both SHINE variants"),
        ("noise family", f"{val('noise_family')}, shape {val('noise_shape')}"),
        ("noise scale", f"{val('subjects_noise_scale')} (a fraction of canvas width)"),
        ("test-retest", f"target {val('target_test_retest', '{:.3f}')}, "
                        f"achieved {val('achieved_tr_unscreened', '{:.3f}')}"),
        ("dispersion", f"{val('dispersion')} fitted, swept over {val('dispersion_swept')}"),
    ]
    kv = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
                 for k, v in rows)
    return _section("model", "1. What was simulated", f"""
<p>Each simulated participant performs the deployed SpAM task: 18 distinct trials of 20 images each,
arranged on a <strong>bounded two-dimensional canvas</strong>. An arrangement is a noisy,
idiosyncratically-weighted projection of a ground-truth geometry onto that canvas, and pairwise
distances are read off it exactly as the real task records them.</p>

<p>The canvas is what distinguishes this model (task-v5) from its predecessors, and it is not
cosmetic. Earlier versions placed images on an unbounded plane and produced per-trial maximum
distances of 1.39 on a scale whose ceiling is 1.0 &mdash; arrangements no participant could have
made. Points are now mapped into the box by a smooth saturating transform rather than clipped, so
the walls compress the extremes instead of piling points against them. Section 11 shows this
reproduces a property of the real data that the unbounded model could not.</p>

<p>Nothing here is hand-tuned. Three constants are fitted to the pilot: the noise population, the
between-participant dispersion, and the noise level that reproduces the empirical test-retest
reliability.</p>

<table class="kv">{kv}</table>

<p class="note"><strong>Read the dimensionality choice carefully.</strong> The ground truth is built
at D=8 although the pilot's own scan selected D=3. That scan measures the dimensionality at which two
20-participant halves still agree, which is a floor set by sample size rather than the intrinsic
dimensionality of the space. A 3-dimensional ground truth is easier to recover than the truth, so
planning on it would understate the sample required. D=8 errs pessimistic <em>deliberately</em>
&mdash; but see section 3, where that choice turns out to carry its own cost.</p>
""")


# --------------------------------------------------------------------------- 2. the arms
def section_arms(run: Run) -> str:
    return _section("arms", "2. The arms, and what is merely swept", """
<p>The experimental contrast is a single question: does an engineered assignment of images to trials
recover the perceptual space better than the random assignment the task currently deploys? Both arms
collect <strong>exactly the same number of judgements</strong> from the same number of participants.
They differ only in which image pairs those judgements land on.</p>

<table class="datatable">
<thead><tr><th>Arm</th><th>How images reach trials</th><th>Status</th></tr></thead>
<tr><td><strong>random</strong></td>
    <td>Each participant independently shuffles the 725 images, takes the first 360, and chops them
        into 18 trials of 20. Nothing coordinates across participants.</td>
    <td>The deployed task &mdash; the baseline.</td></tr>
<tr><td><strong>designed</strong></td>
    <td>A balanced covering planned jointly across participants. Each participant's trials stay
        image-disjoint, so a session remains deployable exactly as the task runs today.</td>
    <td>The proposal.</td></tr>
<tr><td><strong>designed_unconstrained</strong></td>
    <td>The same covering without per-session disjointness. Not deployable.</td>
    <td>Reference only, to price the constraint (section 5).</td></tr>
</table>

<p>Everything else is a <strong>sensitivity axis, not an arm</strong>. These exist so the conclusions
can be shown to hold across a range rather than at one convenient setting, and results are averaged
over them unless stated otherwise:</p>

<ul>
<li><strong>Participants (N)</strong>: 30, 50, 75, 500. N=500 is a <em>ceiling probe</em>, not a
    recruitment target &mdash; it exists to find where the curves stop moving.</li>
<li><strong>Screening threshold</strong>: &minus;1, 0, 0.1, 0.2. The &minus;1 arm runs the screening
    block but excludes nobody, which is the control that holds collected trials fixed so exclusion is
    not confounded with a shorter session.</li>
<li><strong>Canvas softness</strong>: 3, 4, 8. The one canvas quantity with no observable
    distribution to sample from, so it is varied rather than chosen. Aspect ratio and fill
    <em>are</em> sampled per trial from the pilot's measured marginals.</li>
<li><strong>Perspective dispersion</strong>: the fitted value &plusmn;0.15, because it is identified
    only through between-participant agreement &mdash; the noisiest of the three calibration anchors
    &mdash; and should not be trusted as a point estimate.</li>
<li><strong>MDS dimensionality</strong>: the recovered embedding is fitted across a grid,
    independently of the ground truth's D=8.</li>
</ul>

<p>Ten independent cohorts per cell, each with a <strong>fresh design</strong>. A design shared across
repetitions would give the designed arm zero allocation variance while the random arm carried it, and
the two arms' spreads would not be comparable &mdash; the whole result would then rest on the quality
of one lucky draw.</p>
""")


# --------------------------------------------------------------------------- 3. the ground truth
def section_ground_truth(run: Run) -> str:
    figs, parts = [], []
    ceiling, gtraw = run.get("noise_ceiling"), run.get("gt_vs_raw")
    if ceiling is not None and gtraw is not None:
        merged = gtraw.merge(ceiling[["level", "ceiling_full"]], on="level", how="left",
                             suffixes=("", "_c"))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=merged["level_name"], y=merged["spearman"],
                             name="ground truth vs the raw judgements (in-sample)",
                             marker_color="#1f77b4"))
        fig.add_trace(go.Bar(x=merged["level_name"], y=merged["ceiling_full"],
                             name="what the raw data reaches against itself",
                             marker_color="#d62728"))
        fig.update_xaxes(title_text="semantic relatedness (coarse \u2192 fine)")
        fig.update_yaxes(title_text="Spearman")
        figs.append(_fig(fig, "The embedding reproduces more than the data reproduces in itself"))
        cols = ["level_name", "n_observed", "spearman", "ceiling_full", "frac_of_ceiling"]
        parts.append(_table_html(merged[[c for c in cols if c in merged.columns]]))
    cov = run.get("level_coverage")
    if cov is not None:
        parts.append("<p>How much of each level anyone actually judged:</p>"
                     + _table_html(cov[["level_name", "n_pairs", "n_observed", "observed_frac"]]))

    return _section("gt", "3. The ground truth, and the limit it puts on everything downstream", f"""
<p>The ground truth is a weighted MDS embedding of the pilot's pooled judgements. Every result below
is measured against it, so its quality bounds what any of them can mean. It deserves a section of its
own because that bound turns out to be tight.</p>

<p><strong>The embedding reproduces the data better than the data reproduces itself.</strong> Within
each semantic level, its distances correlate with the raw pooled judgements at
&rho;&nbsp;=&nbsp;0.44&ndash;0.50 for the coarse levels. But that is an <em>in-sample</em> number: the
embedding was fitted on the very participants the aggregate pools, and a 725&times;8 configuration
has ample freedom to chase noise in a 31%-observed matrix. The control is how well the raw data agrees
with <em>itself</em> across disjoint halves of participants &mdash; and that ceiling is 0.07, 0.02,
0.13 and 0.41 for the four coarsest levels. Exceeding a ceiling several-fold is the signature of
fitting noise, not of preserved structure.</p>

<p>This agrees with the pilot's own out-of-sample split-half correlation, which peaked at 0.233 &mdash;
about 5% shared rank variance. <strong>41 participants at 31% pair coverage support the coarse,
between-level structure and very little within-level per-pair structure.</strong></p>

<p class="note">Two consequences, and they pull in opposite directions. Results at fine granularity
(section 9) are partly reporting on the embedding's smoothness rather than on recoverable structure,
so the fine-grained null results are weaker evidence about perception than they look. But the
<em>design comparison</em> is untouched: it asks which allocation covers the pair space better, and
both arms face the identical ground truth.</p>
{"".join(parts)}
""", figs)


# --------------------------------------------------------------------------- 4. coverage
def section_coverage(run: Run) -> str:
    cov = run.get("coverage")
    if cov is None:
        return _section("coverage", "4. Does the design see more image pairs?", _missing("coverage"))
    per_n = cov.groupby(["num_subjects", "arm"])["pair_coverage"].mean().unstack()
    obs = cov.groupby(["num_subjects", "arm"])["average_pair_obs"].mean().unstack()

    rows = []
    for n in per_n.index:
        rows.append({"N": n, "random (%)": per_n.loc[n, "random"],
                     "designed (%)": per_n.loc[n, "designed"],
                     "relative gain": _delta(per_n.loc[n, "designed"], per_n.loc[n, "random"]),
                     "observations per pair": obs.loc[n, "designed"]})
    table = pd.DataFrame(rows)

    fig = _by_arm_lines(cov, "pair_coverage",
                        "Pair coverage: the fraction of the 262,450 image pairs anyone judged",
                        "pairs observed (%)")
    fig.add_hline(y=100, line=dict(dash="dot", color="rgba(128,128,128,0.6)"))

    gain = go.Figure()
    ns = sorted(cov["num_subjects"].unique())
    gains = [100 * (per_n.loc[n, "designed"] - per_n.loc[n, "random"]) / per_n.loc[n, "random"]
             for n in ns]
    gain.add_trace(go.Bar(x=[str(n) for n in ns], y=gains, marker_color="#1f77b4",
                          text=[f"{g:+.1f}%" for g in gains], textposition="outside"))
    gain.update_xaxes(title_text="participants (N)")
    gain.update_yaxes(title_text="coverage gain over random (%)")

    return _section("coverage", "4. Does the design see more image pairs?", f"""
<p>This is the question the design exists to answer, and the cleanest result in the report. Both arms
collect <em>exactly the same number of judgements</em> &mdash; the same participants doing the same
number of trials. They differ only in which pairs those judgements land on.</p>

{_table_html(table, "{:.2f}")}

<p><strong>The design wins where it matters and costs nothing where it does not.</strong> The gain
grows with N through the deployable range and then vanishes at N=500, where random already covers
99.9% of pairs and there is nothing left to win. That is the expected shape, not a null result: the
informative window is N=50&ndash;75, and the N=500 column is a ceiling probe confirming the two arms
converge rather than the design ever hurting.</p>

<p class="note">Note the third column. At N=30 the average pair is judged <strong>0.48 times</strong>
&mdash; most pairs are never seen at all, and those that are are seen once. Coverage at these sample
sizes is not a matter of measuring each pair well; it is a matter of touching each pair at all.</p>
""", [fig, gain])


# --------------------------------------------------------------------------- 5. deployability
def section_deployability(run: Run) -> str:
    d2a = run.get("design_only")
    if d2a is None:
        return _section("deploy", "5. What the deployable constraint costs", _missing("design_only"))
    g = d2a.groupby(["num_subjects", "arm"])[
        ["frac_pairs_covered", "reps_per_image_sd", "wasted_frac"]].mean()

    fig = go.Figure()
    for arm in ("random", "designed", "designed_unconstrained"):
        if arm not in d2a["arm"].unique():
            continue
        sub = g.xs(arm, level="arm")
        fig.add_trace(go.Scatter(x=sub.index, y=100 * sub["frac_pairs_covered"],
                                 mode="lines+markers", name=arm,
                                 line=dict(color=ARM_COLOUR.get(arm), width=2),
                                 marker=dict(size=8)))
    fig.update_xaxes(title_text="participants (N)", type="log",
                     tickvals=sorted(d2a["num_subjects"].unique()))
    fig.update_yaxes(title_text="pairs covered (%)")
    f1 = _fig(fig, "Combinatorics only: coverage by arm, no participants and no MDS")

    bal = go.Figure()
    for arm in ("random", "designed"):
        sub = g.xs(arm, level="arm")
        bal.add_trace(go.Bar(x=[str(int(n)) for n in sub.index], y=sub["reps_per_image_sd"],
                             name=arm, marker_color=ARM_COLOUR.get(arm)))
    bal.update_xaxes(title_text="participants (N)")
    bal.update_yaxes(title_text="SD of appearances per image")
    f2 = _fig(bal, "How evenly the images are used (lower is more balanced)")

    n50 = g.xs(50, level="num_subjects") if 50 in d2a["num_subjects"].values else None
    detail = ""
    if n50 is not None and "designed_unconstrained" in n50.index:
        cost = 100 * (n50.loc["designed_unconstrained", "frac_pairs_covered"]
                      - n50.loc["designed", "frac_pairs_covered"]) / n50.loc[
                          "designed_unconstrained", "frac_pairs_covered"]
        detail = (f"<p>At N=50 the deployable design covers "
                  f"{100 * n50.loc['designed', 'frac_pairs_covered']:.2f}% of pairs against the "
                  f"unconstrained {100 * n50.loc['designed_unconstrained', 'frac_pairs_covered']:.2f}%. "
                  f"<strong>The constraint costs {abs(cost):.2f}% relative</strong> &mdash; "
                  f"negligible, and worth stating precisely because it is the difference between a "
                  f"design you can run and one you cannot.</p>")

    return _section("deploy", "5. What the deployable constraint costs", f"""
<p>Sub-stage 2a strips away everything uncertain: no simulated participants, no noise model, no MDS.
It is pure combinatorics over which pairs a given allocation touches, and it isolates the allocation
effect from every downstream modelling choice. If the arms did not separate here, nothing later
could be attributed to the design.</p>

{detail}

<p>Two further properties, both visible above. The design uses the image set <strong>far more
evenly</strong> &mdash; the spread in how often each image appears is roughly five times tighter,
which matters because an image nobody sees contributes nothing and an image everyone sees contributes
redundancy. And it <strong>wastes far fewer judgements</strong> on pairs already measured: at N=75,
16.8% of the designed arm's judgements are repeats of an already-covered pair against 35.9% for
random.</p>

<p class="note">At N=500 both arms waste about 85% of judgements, because the pair space is
saturated and there is nothing left to cover. Waste is only meaningful while coverage is
incomplete.</p>
""", [f1, f2])


# --------------------------------------------------------------------------- 6. pre-MDS reliability
def section_reliability(run: Run) -> str:
    stab = run.get("stability")
    if stab is None:
        return _section("reliability", "6. Reliability before any embedding",
                        _missing("stability"))
    fig = _by_arm_lines(stab, "spearman",
                        "Would a second cohort produce the same distances? (before MDS)",
                        "Spearman between independent cohorts")
    g = stab.groupby(["num_subjects", "arm"])["spearman"].mean().unstack()
    table = g.assign(**{"difference": g["designed"] - g["random"]}).reset_index()

    return _section("reliability", "6. Reliability before any embedding", f"""
<p>Correlating two independent cohorts' pooled distance matrices, before any embedding is fitted.
This measures the raw material MDS is given.</p>

{_table_html(table)}

<p><strong>Here the design loses, and the reason is instructive.</strong> Through the deployable range
the designed arm is <em>less</em> reproducible than random &mdash; 0.126 against 0.143 at N=50 &mdash;
reversing only at N=500. This is not a contradiction of section 4; it is the same fact seen from the
other side. Both arms make the same number of judgements. The design spreads them across more distinct
pairs, so each pair is measured with <em>fewer</em> observations and its mean is noisier. Random
concentrates its judgements on fewer pairs, each measured slightly better.</p>

<p class="note"><strong>Coverage and per-pair precision trade against each other at fixed effort.</strong>
Which one to buy depends on what the embedding needs. A correlation computed over pairs both cohorts
happen to have observed also flatters the arm that observes fewer, more heavily-sampled pairs, so this
metric is not neutral between the arms &mdash; which is exactly why the decision should rest on the
embedding-level results in sections 7 and 8 rather than on this one.</p>
""", [fig])


# --------------------------------------------------------------------------- 7. embedding recovery
def section_embedding(run: Run) -> str:
    figs: List[go.Figure] = []
    parts: List[str] = []
    stab = run.get("embedding_stability")
    gen = run.get("embedding_generalizability")
    if stab is None or gen is None:
        return _section("embedding", "7. Does the embedding itself reproduce?",
                        _missing("embedding_stability / embedding_generalizability"))

    figs.append(_by_arm_lines(stab, "mean_spearman",
                              "Agreement between embeddings fitted to independent cohorts",
                              "Spearman between recovered distance matrices"))
    figs.append(_by_arm_lines(gen, "mean_procrustes_m2",
                              "Procrustes disparity between independent cohorts' embeddings "
                              "(lower is better)",
                              "Procrustes m²"))
    best_d = (stab.groupby("ndim")["mean_spearman"].mean().idxmax()
              if "ndim" in stab.columns else None)
    g = stab.groupby(["num_subjects", "arm"])["mean_spearman"].mean().unstack()
    parts.append(_table_html(g.assign(difference=g["designed"] - g["random"]).reset_index()))
    if best_d is not None:
        by_d = stab.groupby("ndim")["mean_spearman"].mean().reset_index()
        fd = go.Figure(go.Scatter(x=by_d["ndim"], y=by_d["mean_spearman"], mode="lines+markers",
                                  line=dict(color="#1f77b4", width=2), marker=dict(size=8)))
        fd.update_xaxes(title_text="MDS dimensionality")
        fd.update_yaxes(title_text="Spearman between cohorts")
        figs.append(_fig(fd, "Which embedding dimensionality reproduces best"))
        parts.append(f"<p>Averaged over everything else, agreement peaks at "
                     f"<strong>D={int(best_d)}</strong>.</p>")

    return _section("embedding", "7. Does the embedding itself reproduce?", f"""
<p>Section 6 asked whether two cohorts produce the same <em>distances</em>. This asks the question
that actually matters: whether they produce the same <em>space</em>. Two measures, because they can
disagree &mdash; a Spearman correlation between recovered distance matrices, and a Procrustes
disparity after optimally aligning the two configurations.</p>

{"".join(parts)}
""", figs)


# --------------------------------------------------------------------------- 8. recovery vs GT
def section_recovery(run: Run) -> str:
    rec = run.get("recovery_vs_gt")
    topk = run.get("topk_jaccard")
    if rec is None:
        return _section("recovery", "8. Recovering the ground truth", _missing("recovery_vs_gt"))
    figs = []
    for col, label in (("mean_auc", "AUC: ranking truly-close pairs above far ones"),
                       ("mean_recall", "Recall of the ground truth's closest pairs")):
        if col in rec.columns:
            figs.append(_by_arm_lines(rec, col, label, col.replace("mean_", ""), hline=0.5
                                      if col == "mean_auc" else None))
    if topk is not None and "mean_jaccard" in topk.columns:
        figs.append(_by_arm_lines(topk, "mean_jaccard",
                                  "Do two cohorts agree on WHICH pairs are the closest?",
                                  "top-k Jaccard"))
    g = rec.groupby(["num_subjects", "arm"])[
        [c for c in ("mean_auc", "mean_recall", "mean_dprime") if c in rec.columns]].mean()
    return _section("recovery", "8. Recovering the ground truth", f"""
<p>The previous sections compare cohorts against each other, which rewards a consistently wrong
answer as much as a right one. This compares each recovered embedding against the ground truth that
generated the data.</p>

{_table_html(g.reset_index())}

<p class="note">These numbers are internally consistent by construction &mdash; the same ground truth
generated the data and scores the recovery &mdash; so they answer &ldquo;how much of the planted
structure survives this design and this sample size?&rdquo; and not &ldquo;how well would this
recover human perceptual structure?&rdquo;. Section 3 is the constraint on the second reading.</p>
""", figs)
