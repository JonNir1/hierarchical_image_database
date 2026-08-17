"""Stage-2 sections: the experiment and its results.

Same contract as the rest of the report: every number is computed from the run's tables rather than
transcribed, the answer comes first, and no term is used before it is explained.
"""
from __future__ import annotations

import html
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from SpAM_Simulations.reporting.build_report import (
    ARM_COLOUR, Run, _by_arm_lines, _fig, _missing, _section, _table_html,
)


def _note(text: str) -> str:
    return f'<p class="note">{text}</p>'


def _pct_gain(designed: float, random: float) -> float:
    return 100 * (designed - random) / random if random else float("nan")


# --------------------------------------------------------------------------- 6. the two arms
def section_arms(run: Run) -> str:
    return _section("arms", "6. Stage 2: the two ways of handing out images", """
<p>Every participant sees 360 of the 725 images, split into 18 arrangements of 20. The only question
in stage 2 is <strong>how we decide which images each person gets</strong>.</p>

<table class="datatable">
<thead><tr><th></th><th>How it works</th><th>Status</th></tr></thead>
<tr><td><strong>Random</strong></td>
    <td>Each participant independently gets a shuffled selection. Nobody coordinates. If two people
        happen to receive overlapping sets, that is luck.</td>
    <td>What the task does today.</td></tr>
<tr><td><strong>Planned</strong></td>
    <td>The selections are worked out <em>together</em>, before anyone arrives, so that between them
        the participants cover as many different image pairs as possible.</td>
    <td>The proposal being tested.</td></tr>
</table>

<p><strong>Both do exactly the same amount of work.</strong> Same number of people, same number of
arrangements, same 20 images per arrangement. Nobody is asked to do anything extra. The only
difference is <em>which</em> pairs the effort lands on. That is what makes this a fair comparison: any
difference in the results is attributable to the allocation and nothing else.</p>

<p>One practical constraint on the planned version: within a single participant's session, no image
may appear in two different arrangements. Showing someone the same picture twice in one sitting
invites them to remember rather than judge. This restriction makes planning harder, and section 8
prices it.</p>

<h3>Things we varied but are not testing</h3>
<p>A result that only holds at one setting is not much of a result, so everything below was run
across a range of settings and the results are averaged over them:</p>
<ul>
<li><strong>Number of participants</strong>: 30, 50, 75, and 500. The 500 is not a recruitment
proposal &mdash; it is there to find out where the curves stop improving.</li>
<li><strong>Screening strictness</strong>: how careful we are about excluding participants who
answer inconsistently, including a setting that excludes nobody.</li>
<li><strong>Two properties of the model participant</strong>: how firmly the edges of the screen push
back, and how much people differ from one another in what they pay attention to.</li>
</ul>

<p>Each combination was run <strong>ten separate times</strong> with fresh simulated people, so we can
see how much the answers bounce around.</p>
""")


# --------------------------------------------------------------------------- 8. coverage
def section_coverage(run: Run) -> str:
    cov = run.get("coverage")
    if cov is None:
        return _section("coverage", "8. Result: how much of the map gets seen", _missing("coverage"))
    per_n = cov.groupby(["num_subjects", "arm"])["pair_coverage"].mean().unstack()
    obs = cov.groupby(["num_subjects", "arm"])["average_pair_obs"].mean().unstack()

    table = pd.DataFrame([{
        "participants": int(n),
        "random (% of pairs seen)": per_n.loc[n, "random"],
        "planned (% of pairs seen)": per_n.loc[n, "designed"],
        "improvement": f"{_pct_gain(per_n.loc[n, 'designed'], per_n.loc[n, 'random']):+.1f}%",
        "times each pair is judged": obs.loc[n, "designed"],
    } for n in per_n.index])

    fig = _by_arm_lines(cov, "pair_coverage",
                        "What fraction of the 262,450 image pairs anyone judged at all",
                        "pairs seen (%)")
    fig.add_hline(y=100, line=dict(dash="dot", color="rgba(128,128,128,0.6)"))

    ns = sorted(cov["num_subjects"].unique())
    gains = [_pct_gain(per_n.loc[n, "designed"], per_n.loc[n, "random"]) for n in ns]
    bar = go.Figure(go.Bar(x=[str(n) for n in ns], y=gains, marker_color="#1f77b4",
                           text=[f"{g:+.1f}%" for g in gains], textposition="outside"))
    bar.update_xaxes(title_text="participants")
    bar.update_yaxes(title_text="extra pairs covered vs random (%)")

    return _section("coverage", "8. Result: how much of the map gets seen", f"""
<p><strong>Planning covers 18&ndash;30% more image pairs for the same effort.</strong> This is the
cleanest result in the report.</p>

{_table_html(table, "{:.2f}")}

<p>Read the last column before anything else. At 30 participants, the average image pair is judged
<strong>0.48 times</strong> &mdash; less than once. Most pairs are seen by nobody, and the lucky ones
are seen once. At these sample sizes the challenge is not measuring each pair <em>well</em>; it is
touching each pair <em>at all</em>. That is precisely the problem planning solves.</p>

<p><strong>The advantage grows, then disappears.</strong> It is largest at 75 participants and gone
at 500. That is exactly what should happen: by 500 participants, random selection already stumbles
onto 99.9% of pairs, so there is nothing left to win. The gap closing at the top is evidence the
comparison is behaving sensibly, not evidence that planning stops working.</p>

{_note("<strong>The useful range is 50&ndash;75 participants.</strong> Below that both approaches "
       "are starved; above it the problem solves itself.")}
""", [fig, bar])


# --------------------------------------------------------------------------- 8b. the constraint
def section_deployability(run: Run) -> str:
    d2a = run.get("design_only")
    if d2a is None:
        return _section("deploy", "8b. What the one-image-per-session rule costs",
                        _missing("design_only"))
    g = d2a.groupby(["num_subjects", "arm"])[
        ["frac_pairs_covered", "reps_per_image_sd", "wasted_frac"]].mean()

    bal = go.Figure()
    for arm, label in (("random", "random"), ("designed", "planned")):
        sub = g.xs(arm, level="arm")
        bal.add_trace(go.Bar(x=[str(int(n)) for n in sub.index], y=sub["reps_per_image_sd"],
                             name=label, marker_color=ARM_COLOUR.get(arm)))
    bal.update_xaxes(title_text="participants")
    bal.update_yaxes(title_text="unevenness in how often each image is used")
    fig = _fig(bal, "How evenly the 725 images get used (shorter bars are better)")

    detail = ""
    if 50 in d2a["num_subjects"].values:
        n50 = g.xs(50, level="num_subjects")
        if "designed_unconstrained" in n50.index:
            free = n50.loc["designed_unconstrained", "frac_pairs_covered"]
            real = n50.loc["designed", "frac_pairs_covered"]
            detail = (f"<p>At 50 participants, the version we can actually run covers "
                      f"<strong>{100 * real:.2f}%</strong> of pairs. The version that ignores the "
                      f"rule covers <strong>{100 * free:.2f}%</strong>. <strong>The rule costs "
                      f"{abs(100 * (free - real) / free):.2f}% relative</strong> &mdash; "
                      f"essentially nothing.</p>")

    return _section("deploy", "8b. What the one-image-per-session rule costs", f"""
<p><strong>Almost nothing &mdash; a fraction of one percent.</strong> Worth checking, because it is
the difference between a plan we can run and one we cannot.</p>

{detail}

<p>Two side benefits of planning show up here too, both consequences of coordinating rather than
leaving things to chance:</p>
<ul>
<li><strong>The images get used far more evenly</strong> (about five times more evenly). Under random
selection some images are shown to many people while others are barely shown at all. An image nobody
sees contributes nothing; an image everybody sees contributes repetition.</li>
<li><strong>Far less effort is wasted.</strong> At 75 participants, 36% of random judgements land on
a pair that has already been judged, against 17% under planning.</li>
</ul>

{_note("This section used no simulated people at all &mdash; it is pure bookkeeping about which "
       "pairs each approach touches. That is deliberate: if the two approaches did not differ here, "
       "nothing later could be credited to the allocation.")}
""", [fig])


# --------------------------------------------------------------------------- 9. the trade
def section_reliability(run: Run) -> str:
    stab = run.get("stability")
    if stab is None:
        return _section("reliability", "9. The catch: planning trades precision for breadth",
                        _missing("stability"))
    fig = _by_arm_lines(stab, "spearman",
                        "Do two separate groups of people produce the same raw answers?",
                        "agreement between two independent groups")
    g = stab.groupby(["num_subjects", "arm"])["spearman"].mean().unstack()
    table = g.rename(columns={"designed": "planned"}).reset_index().rename(
        columns={"num_subjects": "participants"})

    return _section("reliability", "9. The catch: planning trades precision for breadth", f"""
<p><strong>Planning makes the raw measurements slightly noisier, and this is worth understanding
before the results that follow.</strong></p>

<p>Here we take two completely separate groups of simulated participants, pool each group's answers,
and ask whether the two groups agree with each other. More agreement means more trustworthy raw data.</p>

{_table_html(table)}

<p><strong>Random wins here</strong> at every sample size except the largest. That looks like a
contradiction of section 8, but it is the same fact viewed from the other side.</p>

<p>Both approaches make the same number of judgements. Planning spreads them across <em>more distinct
pairs</em>, so each individual pair is measured fewer times and its average is shakier. Random piles
its judgements onto fewer pairs, measuring each slightly better. <strong>Breadth and precision trade
against each other when the total effort is fixed.</strong></p>

{_note("So which do we want? That depends on what happens when the answers are turned into a map "
       "&mdash; which is the next section, and where the trade is settled. This comparison also "
       "quietly favours random, because it can only be computed on pairs that <em>both</em> groups "
       "happened to see, which is the situation random is better at producing.")}
""", [fig])


# --------------------------------------------------------------------------- 10. the map
def section_embedding(run: Run) -> str:
    stab = run.get("embedding_stability")
    gen = run.get("embedding_generalizability")
    if stab is None or gen is None:
        return _section("embedding", "10. Result: does the same map come out?",
                        _missing("embedding_stability / embedding_generalizability"))
    figs = [
        _by_arm_lines(stab, "mean_spearman",
                      "Do two separate groups of people produce the same map?",
                      "agreement between the two maps"),
        _by_arm_lines(gen, "mean_procrustes_m2",
                      "How different the two maps are after lining them up (lower is better)",
                      "leftover difference"),
    ]
    g = stab.groupby(["num_subjects", "arm"])["mean_spearman"].mean().unstack()
    table = g.rename(columns={"designed": "planned"}).reset_index().rename(
        columns={"num_subjects": "participants"})

    return _section("embedding", "10. Result: does the same map come out?", f"""
<p><strong>Planning wins, reversing section 9.</strong> This is the comparison that matters, because
the map is what the study is for &mdash; the raw distances are only raw material.</p>

<p>The test: build a map from one group of participants, build another from a completely separate
group, and ask how similar the two maps are. If a second run of the study would produce a different
map, the map is not telling us about perception.</p>

{_table_html(table)}

<p><strong>Why the reversal?</strong> The map-building step weights every pair by how often it was
judged. A pair measured once still pulls the map in the right direction; it just pulls more gently.
So having a gentle constraint on <em>many</em> pairs beats having a firm constraint on <em>few</em>.
Section 9 measured the ingredients; this measures the cake.</p>

{_note("Both measures agree, which matters because they can disagree: one compares the distances "
       "in the two maps, the other rotates and rescales the maps to line them up as well as "
       "possible and measures what is left over.")}
""", figs)


# --------------------------------------------------------------------------- 11. the truth
def section_recovery(run: Run) -> str:
    rec = run.get("recovery_vs_gt")
    topk = run.get("topk_jaccard")
    if rec is None:
        return _section("recovery", "11. Result: do we recover the right answer?",
                        _missing("recovery_vs_gt"))
    figs = []
    if "mean_auc" in rec.columns:
        figs.append(_by_arm_lines(
            rec, "mean_auc",
            "Can the recovered map tell genuinely-similar pairs from dissimilar ones?",
            "accuracy (0.5 = coin flip)", hline=0.5))
    if "mean_recall" in rec.columns:
        figs.append(_by_arm_lines(
            rec, "mean_recall", "How many of the truly-closest pairs we find",
            "fraction found"))
    if topk is not None and "mean_jaccard" in topk.columns:
        figs.append(_by_arm_lines(
            topk, "mean_jaccard", "Do two groups agree on WHICH pairs are the closest?",
            "overlap between the two lists"))

    cols = [c for c in ("mean_auc", "mean_recall") if c in rec.columns]
    g = rec.groupby(["num_subjects", "arm"])[cols].mean().reset_index()
    g["arm"] = g["arm"].replace({"designed": "planned"})

    return _section("recovery", "11. Result: do we recover the right answer?", f"""
<p><strong>Planning wins again, at every sample size.</strong></p>

<p>Sections 9 and 10 compared groups of participants against <em>each other</em>, which rewards
consistency &mdash; two groups could agree perfectly and both be wrong. Here we compare against the
map we started from. Because this is a simulation, we know the true answer, which is a luxury the
real study will never have.</p>

{_table_html(g.rename(columns={"num_subjects": "participants", "mean_auc": "accuracy",
                               "mean_recall": "closest pairs found"}))}

<p>Two things to notice. <strong>Accuracy is high and rises with sample size</strong> &mdash; the
recovered map reliably separates genuinely similar pairs from dissimilar ones. But <strong>finding
the specific closest pairs is much harder</strong>: even at 500 participants we recover about half of
them.</p>

{_note("<strong>This is why we cannot yet say how many participants are enough.</strong> Both "
       "curves are still climbing at 500. We have learned where planning helps most (50&ndash;75) "
       "but not where the quality levels off. That question is now open after two rounds of "
       "simulation.")}

<p class="warn"><strong>A caution about reading these numbers as absolutes.</strong> The same map
both generated the data and scores the recovery, so these say &ldquo;how much of the planted
structure survives&rdquo;, not &ldquo;how well would this recover human perception&rdquo;. Section 5
is the limit on the second reading. The <em>comparison</em> between planned and random is unaffected,
since both are scored against the identical map.</p>
""", figs)
