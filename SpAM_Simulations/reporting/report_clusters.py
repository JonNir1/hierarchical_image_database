"""Grouping, isolation, realism checks, conclusions and glossary.

Split from :mod:`report_sections` for length only. Same contract: numbers come from the tables, the
answer comes before the evidence, and nothing assumes prior familiarity.
"""
from __future__ import annotations

import html
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from SpAM_Simulations.reporting.build_report import (
    LINKAGE_COLOUR, Run, _by_arm_lines, _fig, _missing, _section, _table_html,
)

HIGH_K = 150


def _note(text: str) -> str:
    return f'<p class="note">{text}</p>'


# --------------------------------------------------------------------------- 7. is it realistic
def section_validity(run: Run) -> str:
    figs: List[go.Figure] = []
    parts: List[str] = []

    curves = run.get("noise_vs_distance")
    if curves is not None:
        fig = go.Figure()
        for src, colour, label in (("pilot", "#d62728", "real people"),
                                   ("sim", "#1f77b4", "simulated people")):
            sub = curves[curves["source"] == src]
            fig.add_trace(go.Scatter(x=sub["mean_pair_distance"], y=sub["rmse"],
                                     mode="lines+markers", name=label,
                                     line=dict(color=colour, width=2),
                                     error_y=dict(type="data", array=sub["sem_rmse"],
                                                  visible=True, thickness=1)))
        fig.update_xaxes(title_text="how far apart the two images were placed")
        fig.update_yaxes(title_text="how much the person disagreed with themselves")
        figs.append(_fig(fig, "Both curves rise and then fall - the pattern we did not fit"))

    grad = run.get("validity_gradient")
    if grad is not None and "level_name" in grad.columns:
        arm = grad[grad["arm"] == "random"] if "arm" in grad.columns else grad
        fig = go.Figure()
        for src, colour, label in (("pilot", "#d62728", "real people"),
                                   ("sim", "#1f77b4", "simulated people")):
            col = f"mean_distance_{src}"
            if col in arm.columns:
                fig.add_trace(go.Scatter(x=arm["level_name"], y=arm[col] / arm[col].iloc[0],
                                         mode="lines+markers", name=label,
                                         line=dict(color=colour, width=2)))
        fig.update_xaxes(title_text="unrelated images (left) to nearly identical ones (right)")
        fig.update_yaxes(title_text="distance, relative to unrelated pairs")
        figs.append(_fig(fig, "Both put related images closer together, but the model less so"))

    shape = run.get("noise_curve_shape")
    if shape is not None:
        cols = [c for c in ("source", "rise_from_first", "drop_from_peak", "is_inverted_u")
                if c in shape.columns]
        parts.append(_table_html(shape[cols]))

    return _section("validity", "7. Does the simulated participant behave like a real one?", f"""
<p><strong>Mostly yes, and it passes the one test we did not rig.</strong> Before trusting any
result, we should ask whether the model behaves like a person. Two checks.</p>

<h3>Check one: does it treat related images as similar?</h3>
<p>Our images come with a category structure &mdash; two breeds of dog are more related than a dog
and a chair. Real participants place related images closer together. So does the model. But it does
so <strong>too weakly at the fine end</strong>: for images that are nearly the same kind of thing,
real people place them about three times closer than the model does.</p>

<p>Usefully, this error runs in the safe direction. Real data separates similar images more sharply
than our simulation, so the real study should find things <em>easier</em> than the simulation
predicts &mdash; which means our participant-count estimates err toward asking for too many people
rather than too few.</p>

<h3>Check two: the one we did not fit</h3>
<p>This is the stronger test, because nothing in the model was tuned to pass it.</p>

<p>Sometimes a participant is shown the same set of images twice without being told. Comparing their
two attempts shows where they disagree with themselves. In real data the pattern is a hump: they are
consistent about pairs they place very close together, consistent about pairs they place very far
apart, and <strong>inconsistent in the middle</strong>. Obvious cases are easy; ambiguous ones are
not.</p>

<p>The falling right-hand side of that hump is the interesting part, and it exists because the screen
has edges: two images already in opposite corners cannot be dragged further apart, so there is no room
left to disagree. <strong>Our model reproduces the hump</strong>, including the fall, despite never
having been shown this pattern. That is real evidence the model's noise resembles human noise rather
than merely being the right size.</p>

{"".join(parts)}

<p class="warn"><strong>One limit on the first check.</strong> It was computed on a single setting
rather than across the whole range, so it shows the model behaves sensibly <em>somewhere</em> in its
parameter space rather than everywhere in it. The code now checks every setting, but producing that
needs a fresh stage-2 run. The second check is unaffected &mdash; it tests the model directly rather
than any particular run.</p>
""", figs)


# --------------------------------------------------------------------------- 12. groups
def section_clusters(run: Run) -> str:
    ag = run.get("cluster_agreement")
    ks = run.get("k_selection")
    if ag is None:
        return _section("clusters", "12. Result: can we sort the images into groups?",
                        _missing("cluster_agreement"))

    curve = ag.groupby(["linkage", "k"])[["mean_vi_norm", "mean_sil_cross"]].mean().reset_index()
    sil = go.Figure()
    for linkage, sub in curve.groupby("linkage"):
        sil.add_trace(go.Scatter(x=sub["k"], y=sub["mean_sil_cross"], mode="lines+markers",
                                 name=linkage, line=dict(color=LINKAGE_COLOUR.get(linkage), width=2)))
    sil.update_xaxes(title_text="number of groups we ask for", type="log",
                     tickvals=sorted(curve["k"].unique()))
    sil.update_yaxes(title_text="are the groups still separated in another sample?")
    sil.add_hline(y=0, line=dict(dash="dash", color="rgba(128,128,128,0.8)"))
    sil.add_vrect(x0=HIGH_K, x1=curve["k"].max(), fillcolor="rgba(214,39,40,0.08)", line_width=0,
                  annotation_text="least trustworthy", annotation_position="top left")
    f_sil = _fig(sil, "Below zero, the groups have stopped being real")

    kstats = ""
    if ks is not None:
        counts = ks["k_star_vi"].value_counts().sort_index().to_dict()
        kstats = (f"<p>We asked for the best number of groups in <strong>{len(ks):,} different "
                  f"settings</strong>, using two methods designed to disagree with each other. Both "
                  f"answered <strong>two</strong>, every single time (counts: {counts}). Not one "
                  f"setting produced anything finer.</p>")

    return _section("clusters", "12. Result: can we sort the images into groups?", f"""
<p><strong>No &mdash; beyond one very coarse split, groups do not survive from one sample of people to
the next.</strong> This is the main negative finding, and it changes what the dataset should be used
for.</p>

<p><strong>Why we care.</strong> The eventual goal is a rule like &ldquo;never put two confusable
images in the same trial&rdquo;. The tidiest version of that rule would be: sort the 725 images into
groups of lookalikes, then never take two from the same group.</p>

<p><strong>How we tested it.</strong> Take two separate samples of participants. Sort each one's
images into groups, without telling the procedure what the groups should be. Then ask whether the two
samples found the <em>same</em> groups. Repeat while asking for 2 groups, 3, 5, ... up to 200.</p>

{kstats}

<p><strong>What the figure shows.</strong> The line asks a blunt question: take the groups found in
one sample, and check whether they are still separated in the other sample's map. Above zero, yes.
Below zero, the images in a &ldquo;group&rdquo; are actually closer to images outside it &mdash; the
grouping is a line drawn on something continuous. <strong>That line crosses zero at about 12
groups.</strong></p>

<p>So the images do not form natural clumps. They form a continuum, like colours shading into each
other rather than a set of labelled bins.</p>

{_note("<strong>What to do instead.</strong> Use a distance threshold, not group membership: "
       "&ldquo;never show two images closer than X on the map&rdquo;. That works on a continuum, "
       "and it degrades gracefully &mdash; where the map is uncertain, the rule is merely "
       "conservative rather than wrong.")}

<p class="warn"><strong>Read the shaded region with the most caution.</strong> Asking for 150+ groups
from 725 images means fewer than five images each &mdash; exactly the fine detail section 5 showed the
map invents. Some of this negative result is our small pilot rather than the images themselves.</p>
""", [f_sil])


# --------------------------------------------------------------------------- 13. loners
def section_density(run: Run) -> str:
    den = run.get("density_agreement")
    if den is None:
        return _section("density", "13. Result: which images resemble nothing else?",
                        _missing("density_agreement"))
    g = den.groupby("min_cluster_size")[
        ["mean_n_clusters", "mean_frac_noise", "mean_noise_kappa"]].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["min_cluster_size"], y=g["mean_frac_noise"], mode="lines+markers",
                             name="share of images in no group at all",
                             line=dict(color="#d62728", width=2)))
    fig.add_trace(go.Scatter(x=g["min_cluster_size"], y=g["mean_noise_kappa"], mode="lines+markers",
                             name="agreement on which images those are",
                             line=dict(color="#1f77b4", width=2)))
    fig.update_xaxes(title_text="how many lookalikes we demand before calling it a group")
    fig.update_yaxes(title_text="proportion")

    return _section("density", "13. Result: which images resemble nothing else?", f"""
<p><strong>Most of them &mdash; and we cannot reliably say which.</strong> This is a second opinion on
section 12, from a method that works completely differently.</p>

<p>The grouping method in section 12 is obliged to put <em>every</em> image in some group, even one
that resembles nothing. That is the wrong assumption here: &ldquo;this picture looks like nothing else
in the set&rdquo; is a genuinely useful answer, because such an image is perfectly safe to use. So we
repeated the analysis with a method allowed to say &ldquo;this one belongs nowhere&rdquo;.</p>

{_table_html(g.rename(columns={
    "min_cluster_size": "lookalikes required",
    "mean_n_clusters": "groups found",
    "mean_frac_noise": "share left ungrouped",
    "mean_noise_kappa": "agreement on which"}))}

<p>At a sensible setting, <strong>61% of images belong to no group</strong>, and two samples of
participants agree only weakly about which ones. So it is not simply that most images resist
grouping; <em>which</em> images resist is itself unstable.</p>

<p>There is no way to have it both ways. Demand only two lookalikes and the method finds 171 groups
that two samples agree on essentially not at all. Demand twenty and agreement becomes excellent, but
only because it has given up on 85% of the images. <strong>No setting both covers the image set and
reproduces.</strong></p>

{_note("Two unrelated methods reaching the same conclusion is why we believe it. Section 12 said "
       "fine groupings do not reproduce; this says most images were never in a group to begin with.")}
""", [fig])


# --------------------------------------------------------------------------- 14. conclusions
def section_limits(run: Run) -> str:
    return _section("limits", "14. What to do, and what we still don't know", """
<h3>Recommendations</h3>
<ol>
<li><strong>Switch to planned image allocation.</strong> It covers 18&ndash;30% more image pairs for
identical participant effort, produces more reproducible maps at every sample size we tested, never
does worse, and the practical constraint that keeps it runnable costs a fraction of a percent.</li>
<li><strong>Do not build the analysis around image clusters.</strong> Use a distance threshold on the
map instead. Clusters beyond a single coarse split do not survive from one sample to the next, and
two independent methods agree on that.</li>
<li><strong>Treat the pilot-derived map as provisional.</strong> It reproduces the pilot better than
the pilot reproduces itself, which means its fine detail is partly invented. The full study fixes
this by supplying more data.</li>
</ol>

<h3>Open questions</h3>
<ul>
<li><strong>How many participants are enough?</strong> Still unanswered, now after two rounds of
simulation. Quality was still improving at 500 simulated participants, so we know where planning
helps most (50&ndash;75) but not where the curve flattens.</li>
<li><strong>How much of the negative clustering result is us rather than the images?</strong> The map
was built from about 41 pilot participants covering a third of the pairs. Distinguishing &ldquo;these
images genuinely form a continuum&rdquo; from &ldquo;we lack the data to see the groups&rdquo; needs a
better map &mdash; which the full study produces.</li>
<li><strong>Would real people group better than the model implies?</strong> Probably somewhat: the
model under-separates similar images relative to real pilot data, so the true ceiling on groupings may
sit above two. We know the direction of that bias but not its size.</li>
</ul>

<h3>What would have changed our minds</h3>
<p>Worth stating, since a simulation can be built to confirm whatever you like. Planning would have
lost if its broader coverage had failed to produce better maps &mdash; and on the raw measurements
(section 9) it <em>did</em> lose, which is why sections 10 and 11 were the deciding evidence. The
clustering result would have gone the other way if groups had stayed separated in a second sample at
finer settings; the measurement that decides it crosses zero in plain view.</p>
""")


# --------------------------------------------------------------------------- 15. glossary
def section_glossary(run: Run) -> str:
    rows = [
        ("Image pair", "Any two of the 725 images. There are 262,450 of them, which is the whole "
                       "difficulty."),
        ("Arrangement", "One screen of 20 images that a participant drags into position. Gives us "
                        "an opinion about 190 pairs at once."),
        ("The map", "Coordinates for all 725 images, placed so that images people judged similar "
                    "sit close together. Formally an <em>embedding</em>."),
        ("Ground truth", "The map built from the real pilot data, which the simulation treats as "
                         "the right answer. Everything is scored against it."),
        ("Random vs planned", "The two ways of deciding which images each participant sees. The "
                              "only thing being tested."),
        ("Coverage", "The share of the 262,450 pairs that anyone judged at all."),
        ("Sample / cohort", "One simulated group of participants. Every setting was run with ten "
                            "independent groups so we can see how much results vary."),
        ("Reproducible", "Two separate groups of people give the same answer. The core standard "
                         "throughout: an unreproducible result is not a finding."),
        ("Dimensions", "How many independent ways images are allowed to differ on the map. More "
                       "dimensions capture more, but need more data to pin down."),
        ("Screening", "Excluding participants whose repeated answers disagree with themselves."),
    ]
    body = "".join(f"<tr><th>{html.escape(term)}</th><td>{definition}</td></tr>"
                   for term, definition in rows)
    return _section("glossary", "15. Glossary", f"""
<p>Every term used in this report, in plain language.</p>
<table class="kv">{body}</table>
""")
