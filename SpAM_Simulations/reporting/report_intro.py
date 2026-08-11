"""Background and stage-1 sections of the report.

Written for a reader who has never seen this project. Every term is defined where it first appears,
the answer comes before the evidence, and paragraphs stay short. If a sentence needs the reader to
already know what SMACOF or a condensed distance vector is, it does not belong here.
"""
from __future__ import annotations

import html
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from SpAM_Simulations.reporting.build_report import Run, _fig, _section, _table_html


def _note(text: str) -> str:
    return f'<p class="note">{text}</p>'


# --------------------------------------------------------------------------- 0. the whole thing
def section_summary(run: Run) -> str:
    cal = run.calibration
    return _section("summary", "In one page", f"""
<p><strong>The project.</strong> We have 725 photographs of objects. We want to know how similar
people think they look &mdash; not how similar a computer thinks they are, and not how similar their
dictionary definitions are, but how they <em>look</em> to a person. That gives us a &ldquo;perceptual
map&rdquo; where images people find alike sit close together.</p>

<p><strong>Why we need a map.</strong> Later experiments will show people pairs of images. If two
images look almost identical, showing them together tells us nothing. The map is how we avoid that.</p>

<p><strong>The problem.</strong> 725 images make <strong>262,450 possible pairs</strong>. Nobody can
rate that many. Each participant sees a small slice, and we stitch the slices together.</p>

<p><strong>The question this report answers.</strong> Right now, each participant's slice is chosen
<em>at random</em>. Random slices overlap by luck, and some pairs get missed entirely. We could
instead <em>plan</em> the slices across participants so that between them they cover more of the
262,450 pairs. Is planning actually better? And how many participants do we need either way?</p>

<p><strong>How we answered it without running the study.</strong> We built a computer model of a
participant doing the task, tuned it against {cal.get('n_pilot_sessions', 'our')} real pilot
participants, and then ran the study thousands of times inside the computer &mdash; once with random
slices, once with planned slices, at several sample sizes. This report is what came out.</p>

<p><strong>The three answers, in short:</strong></p>
<ul>
<li><strong>Planning wins.</strong> For the same participant effort it covers 18&ndash;30% more image
pairs, and produces a more reliable map on every measure we checked. Adopt it.</li>
<li><strong>The images do not fall into tidy groups.</strong> We looked hard for clusters and found
that beyond a single coarse split, no grouping survives from one sample of people to the next. So
&ldquo;don't show two images from the same cluster&rdquo; is the wrong rule; a distance threshold is
the right one.</li>
<li><strong>We still cannot say how many participants are enough.</strong> Even at 500 simulated
participants the quality was still improving. We know where planning helps most (50&ndash;75), not
where the curve flattens.</li>
</ul>
""")


# --------------------------------------------------------------------------- 1. the task
def section_task(run: Run) -> str:
    return _section("task", "1. What participants actually do", """
<p>A participant sees a blank rectangle on their screen and a handful of images. Their instruction is
simply: <strong>drag these images around so that similar-looking ones end up near each other.</strong>
When they are happy, the arrangement is saved and they get a fresh set of images.</p>

<p>That is the whole task. It is called the <strong>Spatial Arrangement Method</strong>, and its
appeal is efficiency: dragging 20 images into a layout gives us an opinion about all
<strong>190 pairs</strong> among them at once, rather than asking 190 separate questions.</p>

<p>The measurement is just distance. If a participant drops two images close together, we record a
small number; far apart, a large one. We never ask them to explain, and we never give them
categories to sort into &mdash; whatever structure emerges has to come from where they put things.</p>

<p>Each participant does <strong>18 arrangements of 20 images</strong>, which covers 360 of the 725
images. Two things follow from that, and they drive everything else in this report: one person can
never see most of the images, and the pairs they <em>do</em> judge depend entirely on which images we
put in front of them.</p>
""")


# --------------------------------------------------------------------------- 2. why simulate
def section_why_simulate(run: Run) -> str:
    return _section("why", "2. Why simulate instead of just running the study", """
<p>Because the choices we would have to make first are exactly the ones we cannot make without
data. How many participants? Should the image slices be planned or random? How much of the structure
will survive the noise of people disagreeing with each other?</p>

<p>Running the real study to find out is expensive and only answers the question once. So instead we
built a <strong>model participant</strong>: a piece of code that takes a &ldquo;true&rdquo; map, looks
at it from its own slightly idiosyncratic angle, and drags images onto a rectangle with a realistic
amount of sloppiness. Then we can run 500 participants in a few seconds and repeat the whole study
thousands of times.</p>

<p>The catch is obvious: <strong>a simulation only tells you about the world you simulated.</strong>
That is why a large part of this report is not results at all, but checks on whether the model
behaves like real people (section 8) and whether the map we tuned it against is any good
(section 5).</p>

<p>The work ran in two stages, which is worth holding onto because the report follows them:</p>
<ul>
<li><strong>Stage 1</strong> used the real pilot data to build the &ldquo;true&rdquo; map the
simulation would treat as ground truth (sections 3&ndash;5).</li>
<li><strong>Stage 2</strong> then ran the simulated study on top of that map, comparing planned
against random slices (sections 6 onward).</li>
</ul>
""")


# --------------------------------------------------------------------------- 3. building the map
def section_stage1_map(run: Run) -> str:
    selection = run.stage1.get("selection") or {}
    diag = selection.get("split_diagnostics", {}) if isinstance(selection, dict) else {}
    cal = run.calibration
    missing = ("" if selection else
               '<p class="warn">Stage-1 artifacts were not found beside this run, so this section '
               'describes the method without its numbers. Sync the stage-1 <code>gt/</code> prefix '
               'into a sibling <code>gt-construction-v5/</code> directory to fill them in.</p>')

    facts = []
    if selection:
        facts = [("images placed", 725),
                 ("pilot participants used", cal.get("n_pilot_sessions", "?")),
                 ("dimensions the test chose", selection.get("n_dims", "?")),
                 ("dimensions we actually used", cal.get("n_dims", "?"))]
    kv = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                 for k, v in facts)
    kv_html = f"<table class='kv'>{kv}</table>" if facts else ""

    discard = ""
    if diag.get("discard_rate") is not None:
        discard = (f"<p>One housekeeping detail, stated because it sounds alarming and is not: when "
                   f"splitting participants in half for the test in section 4, "
                   f"<strong>{100 * diag['discard_rate']:.0f}% of splits had to be thrown "
                   f"away</strong>. A split is unusable when one half contains no chain of shared "
                   f"images linking everything together &mdash; you cannot build a single map from "
                   f"two disconnected pieces. Early pilot participants did 10 arrangements rather "
                   f"than 18, so they supply fewer links and discards are commoner than they would "
                   f"be with today's task. This biases nothing so long as kept and discarded halves "
                   f"are similarly well covered, which was checked.</p>")

    return _section("map", "3. Stage 1: turning pilot answers into a map", f"""
<p><strong>What this stage produced:</strong> coordinates for all 725 images, built from what real
people actually did. Everything afterwards is measured against it, so how it was made matters &mdash;
and section 5 asks how far to trust it.</p>

{missing}
{kv_html}

<p><strong>The input.</strong> Every pilot participant's arrangements were pooled into one big table:
for each pair of images, how far apart did people put them on average. Because each person only ever
sees a slice, most of that table is blank &mdash; only about a third of the 262,450 pairs were judged
by anyone at all.</p>

<p><strong>The trick.</strong> A technique called <em>multidimensional scaling</em> takes that patchy
table and finds coordinates for every image such that distances between coordinates match the judged
distances as closely as possible. It is like reconstructing a road map from a table of city-to-city
driving distances &mdash; except two thirds of the table is blank, and the version we use ignores the
blanks rather than guessing at them.</p>

<p>The result is a map where images people consistently placed near each other end up near each
other, and never-judged pairs land wherever everything else implies they should.</p>

{discard}
""")


# --------------------------------------------------------------------------- 4. how many dimensions
def section_dimensions(run: Run) -> str:
    scan = run.stage1.get("scan_summary")
    cv = run.stage1.get("cv_summary")
    selection = run.stage1.get("selection") or {}
    chosen = selection.get("n_dims", "?")
    used = run.calibration.get("n_dims", 8)
    figs: List[go.Figure] = []
    peak_txt, body, jac_note = "low", "", ""

    if isinstance(scan, pd.DataFrame) and {"ndim", "spearman_mean"} <= set(scan.columns):
        best = scan.loc[scan["spearman_mean"].idxmax()]
        peak_txt = f"{best['spearman_mean']:.2f}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=scan["ndim"], y=scan["spearman_mean"], mode="lines+markers",
            name="split the people in half", line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            error_y=dict(type="data", array=scan.get("spearman_sem"), visible=True, thickness=1)))
        if isinstance(cv, pd.DataFrame) and {"ndim", "spearman_mean"} <= set(cv.columns):
            fig.add_trace(go.Scatter(
                x=cv["ndim"], y=cv["spearman_mean"], mode="lines+markers",
                name="hold some people out", line=dict(color="#2ca02c", width=2, dash="dot"),
                marker=dict(size=7)))
        fig.update_xaxes(title_text="number of dimensions in the map")
        fig.update_yaxes(title_text="agreement between two separate halves of the pilot")
        figs.append(_fig(fig, "Two halves of the pilot barely agree, at any map size"))

        if "topk_jaccard_mean" in scan.columns:
            f2 = go.Figure(go.Scatter(x=scan["ndim"], y=scan["topk_jaccard_mean"],
                                      mode="lines+markers",
                                      line=dict(color="#ff7f0e", width=2), marker=dict(size=8)))
            f2.update_xaxes(title_text="number of dimensions in the map")
            f2.update_yaxes(title_text="overlap on which pairs are closest")
            figs.append(_fig(f2, "But agreement about the closest pairs never stops improving"))
            jac_note = ("<p>Now compare the two figures. Overall agreement peaks early and then "
                        "falls, while agreement about <em>which pairs are closest</em> keeps rising "
                        "all the way to 20 dimensions. The two measures point in opposite "
                        "directions, which is itself a sign the data is not rich enough to settle "
                        "the question.</p>")

        cols = [c for c in ("ndim", "spearman_mean", "spearman_sem", "topk_jaccard_mean")
                if c in scan.columns]
        body = _table_html(scan[cols].rename(columns={
            "ndim": "dimensions", "spearman_mean": "half-vs-half agreement",
            "spearman_sem": "uncertainty", "topk_jaccard_mean": "agreement on closest pairs"}),
            "{:.4f}")

    return _section("dims", "4. Stage 1: how many dimensions should the map have?", f"""
<p><strong>The test said {chosen}. We deliberately used {used} instead, and the reason matters.</strong></p>

<p>A road map needs two dimensions: east-west and north-south. A map of how things <em>look</em> may
need more &mdash; one axis for roughly how round something is, another for how colourful, and so on.
We cannot know the right number in advance, so we measure it.</p>

<p><strong>How.</strong> Split the pilot participants into two halves at random. Build a map from
each half separately. If the two maps agree, the data supports that many dimensions; if they
disagree, we are fitting noise. Repeat over many splits and many dimension counts, then take the
smallest number that is not clearly worse than the best &mdash; preferring the simpler map whenever
the evidence cannot tell them apart.</p>

<p><strong>The result: agreement peaked at about {peak_txt}, which is low.</strong> Two independent
halves of our pilot produce noticeably different maps; roughly 5% of the ordering is shared. The
curve is also nearly flat across 3, 4 and 5 dimensions, so the data cannot really distinguish
them.</p>

{body}
{jac_note}

<p>So &ldquo;{chosen} dimensions&rdquo; does not mean perception has {chosen} dimensions. It means
<strong>{chosen} is all this pilot can pin down</strong> &mdash; a limit set by how many people we
have, not by the images.</p>

{_note("<strong>Why we used " + str(used) + " anyway.</strong> A simpler map is easier to recover, "
       "so planning the real study on a " + str(chosen) + "-dimensional map would have flattered "
       "us and led to recruiting too few people. Using " + str(used) + " makes the simulation's job "
       "deliberately harder, so its recommendations err toward caution. Section 5 shows the choice "
       "was not free.")}
""", figs)


# --------------------------------------------------------------------------- 5. is the map good
def section_map_quality(run: Run) -> str:
    figs: List[go.Figure] = []
    body: List[str] = []
    ceiling, gtraw = run.get("noise_ceiling"), run.get("gt_vs_raw")
    if ceiling is not None and gtraw is not None:
        merged = gtraw.merge(ceiling[["level", "ceiling_full"]], on="level", how="left",
                             suffixes=("", "_c"))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=merged["level_name"], y=merged["spearman"],
                             name="how well the map matches the pilot answers",
                             marker_color="#1f77b4"))
        fig.add_trace(go.Bar(x=merged["level_name"], y=merged["ceiling_full"],
                             name="how well the pilot answers match THEMSELVES",
                             marker_color="#d62728"))
        fig.update_xaxes(title_text="image pairs, from unrelated (left) to nearly identical (right)")
        fig.update_yaxes(title_text="agreement")
        figs.append(_fig(fig, "The blue bars should not be able to exceed the red ones"))
        cols = ["level_name", "n_observed", "spearman", "ceiling_full", "frac_of_ceiling"]
        body.append(_table_html(merged[[c for c in cols if c in merged.columns]]))

    return _section("quality", "5. Stage 1 check: is the map actually any good?", f"""
<p><strong>Short answer: it captures the broad strokes and invents the fine detail.</strong> This is
the most important caveat in the report, so it gets its own section.</p>

<p><strong>The test.</strong> Ask two questions and compare the answers.</p>
<ol>
<li>How closely does the map match what the pilot participants actually said?</li>
<li>How closely do the pilot participants match <em>each other</em>? Split them in half and see
whether the two halves agree.</li>
</ol>

<p>The second number is a ceiling. The map is built <em>from</em> those answers, so it cannot
legitimately know more about them than they know about themselves. If the map matches the data better
than the data matches itself, the map has memorised noise.</p>

<p><strong>What we found: the map beats that ceiling by 4 to 20 times.</strong> Within groups of
similar images, the map agrees with the pooled answers at around 0.44&ndash;0.50, while two halves of
the participants agree with each other at 0.02&ndash;0.13. That gap is not a good sign. It is the
signature of a map with more freedom (725 images &times; 8 dimensions) than the data can constrain.</p>

{"".join(body)}

<p><strong>What this does and does not undermine.</strong></p>
<ul>
<li>It <strong>does</strong> mean the fine-grained conclusions later &mdash; especially about clusters
of very similar images &mdash; are partly a statement about having only ~41 pilot participants, not
about perception.</li>
<li>It <strong>does not</strong> affect the main comparison. Planned versus random slices is a
question about which pairs get seen, and both are measured against the identical map. A flawed map
handicaps both equally.</li>
</ul>

{_note("This is fixable, and the full study fixes it. The ceiling is low because 41 people judged "
       "about a third of the pairs. More participants raise it directly.")}
""", figs)
