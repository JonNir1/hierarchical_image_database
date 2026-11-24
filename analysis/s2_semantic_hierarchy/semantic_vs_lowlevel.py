"""
Is an RDM's agreement with the tree semantic, or inherited from low-level (pixel) image similarity?
And are CLIP and WordNet the same semantic surface? (5_semantic_vs_lowlevel.ipynb is the front end.)

Two analyses, both with sem_km (the tree) as the TARGET:

  1. Partial rank correlation (5.1). The Spearman correlation of an RDM with sem_km, raw and after
     residualising both on one or more control RDMs (pixels, CLIP). Rank-transform first, then remove the
     controls by least squares, so the partial is a rank statistic like the raw one. Kendall tau-a, the
     primary agreement measure elsewhere, has no residualisation analogue, hence Spearman here.
  2. Commonality analysis (5.2). The R^2 of sem_km (rank-transformed) on a set of predictor RDMs is split
     into 2^k - 1 components: the part unique to each predictor and the part shared by every subset of
     two or more. The components sum to the full-model R^2. A component can be negative when a predictor
     suppresses another's error variance; that is reported, not clipped. Behaviour as the target is a
     different question (RQ3, s1's 6_decomposition) and is not done here.

Everything is NaN-aware: each quantity is computed on the pairs observed in every RDM it involves.
Uncertainty is an image bootstrap (images are the sampling unit); one bootstrap draw feeds every
quantity of a family, so they are mutually consistent and the cost is paid once.

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import semantic_vs_lowlevel as svl
    svl.partial_correlation(rdms["behav_joint"], rdms["sem_km"], rdms["sens_pre"])
    svl.commonality(rdms["sem_km"], {k: rdms[k] for k in ["clip_pre", "sem_wn", "sens_pre"]})
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import rankdata

from analysis.s2_semantic_hierarchy.common import BEHAV, BEHAV_MDS_SET, REFERENCE, cache_path, pairwise_complete
from analysis.utils import plotting
from analysis.utils.stats import image_bootstrap, percentile_ci, z_ranks

_TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# 5.1 Partial rank correlation
# ---------------------------------------------------------------------------


def _partial_from_ranks(ra: np.ndarray, rb: np.ndarray, controls: list[np.ndarray]) -> float:
    if not controls:
        return float(np.corrcoef(ra, rb)[0, 1])
    design = np.column_stack([np.ones_like(ra), *controls])
    res_a = ra - design @ np.linalg.lstsq(design, ra, rcond=None)[0]
    res_b = rb - design @ np.linalg.lstsq(design, rb, rcond=None)[0]
    return float(np.corrcoef(res_a, res_b)[0, 1])


def partial_correlation(a: np.ndarray, b: np.ndarray, *controls: np.ndarray) -> float:
    """
    Rank-based partial correlation of `a` and `b` holding every RDM in `controls` fixed.

    Computed on the pairs observed in all of them. `a`, `b` and each control are rank-transformed on
    those pairs, then the target and predictor ranks are residualised on all control ranks jointly (least
    squares with an intercept). With no controls this is the plain Spearman rho on the same pairs.
    """
    mask = pairwise_complete(a, b, *controls)
    return _partial_from_ranks(rankdata(a[mask]), rankdata(b[mask]), [rankdata(x[mask]) for x in controls])


@dataclass(frozen=True)
class PartialSpec:
    """One raw-vs-partial comparison: `predictor` against the target, controlling for `controls`."""
    predictor: str
    controls: tuple[str, ...]

    @property
    def control_label(self) -> str:
        return " + ".join(plotting.rdm_label(k) for k in self.controls)


def default_specs(names: Sequence[str], *, pixels: str = "sens_pre", clip: str = "clip_pre") -> list[PartialSpec]:
    """The 5.1 comparisons available among `names`: all against pixels, behavior also against CLIP and both."""
    behavior = (BEHAV, *BEHAV_MDS_SET)
    candidates = ("clip_pre", "clip_post", "sem_wn", *behavior)
    specs = [PartialSpec(p, (pixels,)) for p in candidates if p in names]
    for p in behavior:
        if p in names:
            specs += [PartialSpec(p, (clip,)), PartialSpec(p, (clip, pixels))]
    return specs


def _spec_values(rdms: dict[str, np.ndarray], specs: Sequence[PartialSpec], target: str) -> np.ndarray:
    """(n_specs, 2) array of [raw, partial] Spearman with the target; ranks are shared across specs."""
    ranks: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    masks: dict[tuple[str, ...], np.ndarray | slice] = {}
    out = np.empty((len(specs), 2))
    for i, spec in enumerate(specs):
        involved = [target, spec.predictor, *spec.controls]
        incomplete = tuple(k for k in involved if np.isnan(rdms[k]).any())
        if incomplete not in masks:
            masks[incomplete] = pairwise_complete(*(rdms[k] for k in incomplete)) if incomplete else slice(None)
        mask = masks[incomplete]

        def rk(name: str) -> np.ndarray:
            if (name, incomplete) not in ranks:
                ranks[(name, incomplete)] = rankdata(rdms[name][mask])
            return ranks[(name, incomplete)]

        controls = [rk(k) for k in spec.controls]
        out[i] = [_partial_from_ranks(rk(target), rk(spec.predictor), []),
                  _partial_from_ranks(rk(target), rk(spec.predictor), controls)]
    return out


def partial_table(
    rdms: dict[str, np.ndarray], specs: Sequence[PartialSpec], *, target: str = REFERENCE,
    n_boot: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """Raw and partial Spearman with the target per spec, with image-bootstrap 95% CIs (cached)."""
    names = list(dict.fromkeys([target, *(k for s in specs for k in (s.predictor, *s.controls))]))
    point = _spec_values(rdms, specs, target)
    draws = image_bootstrap(
        [rdms[k] for k in names], lambda *v: _spec_values(dict(zip(names, v)), specs, target).ravel(),
        n_boot=n_boot, seed=seed, cache_path=cache_path(f"s5_partial_boot_{len(names)}rdms_{len(specs)}specs.npz"),
        extra_key=("partial", names, tuple((s.predictor, s.controls) for s in specs)),
    )
    rows = []
    for i, spec in enumerate(specs):
        raw_lo, raw_hi = percentile_ci(draws[:, 2 * i])
        par_lo, par_hi = percentile_ci(draws[:, 2 * i + 1])
        rows.append({
            "predictor": spec.predictor, "controls": spec.control_label,
            "raw": point[i, 0], "raw_lo": raw_lo, "raw_hi": raw_hi,
            "partial": point[i, 1], "partial_lo": par_lo, "partial_hi": par_hi,
            "retained": point[i, 1] / point[i, 0],
        })
    return pd.DataFrame(rows)


def plot_partial(table: pd.DataFrame, *, target: str = REFERENCE) -> go.Figure:
    """Dot plot, one row per comparison: pale marker = raw rho, solid marker = partial rho (95% CI bars)."""
    rows = [f"{plotting.rdm_tick_label(r.predictor)}<br><sub>controlling {r.controls}</sub>" for r in table.itertuples()]
    fig = go.Figure()
    for kind, lighten in (("raw", 0.6), ("partial", 0.0)):
        for r, row in zip(table.itertuples(), rows):
            value, lo, hi = getattr(r, kind), getattr(r, f"{kind}_lo"), getattr(r, f"{kind}_hi")
            marker = plotting.rdm_marker(r.predictor, size=11, lighten=lighten)
            fig.add_trace(go.Scatter(
                x=[value], y=[row], mode="markers", marker=marker, showlegend=False,
                error_x=dict(type="data", symmetric=False, array=[hi - value], arrayminus=[value - lo],
                             color=marker["color"], thickness=1.5),
                hovertemplate=f"{kind}: %{{x:.3f}}<extra>{plotting.rdm_label(r.predictor)}</extra>",
            ))
    fig.update_layout(
        title=f"Agreement with {plotting.rdm_label(target)}: raw (pale) vs partial (solid)"
              "<br><sub>Spearman rho on shared pairs, 95% image-bootstrap CI</sub>",
        xaxis=dict(title="Spearman rho with the tree", zeroline=True), yaxis=dict(autorange="reversed"),
        template=_TEMPLATE, height=160 + 62 * len(table), width=900, margin=dict(l=260),
    )
    return fig


# ---------------------------------------------------------------------------
# 5.2 Incremental R^2 and commonality analysis
# ---------------------------------------------------------------------------


def incremental_r2(target: np.ndarray, predictors: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    R^2 of each predictor alone and of all of them jointly, on z-scored ranks.

    If the joint R^2 is close to the sum of the individual ones, the predictors carry complementary rather
    than redundant information about the target. Computed on the pairs observed in every RDM.
    """
    mask = pairwise_complete(target, *predictors.values())
    zt = z_ranks(target[mask])
    zp = {k: z_ranks(v[mask]) for k, v in predictors.items()}
    ones = np.ones_like(zt)

    def fit(cols: list[str]) -> float:
        X = np.column_stack([zp[c] for c in cols] + [ones])
        beta, *_ = np.linalg.lstsq(X, zt, rcond=None)
        return float(1.0 - np.var(zt - X @ beta) / np.var(zt))

    rows = [{"predictors": k, "r2": fit([k])} for k in zp]
    rows.append({"predictors": " + ".join(zp), "r2": fit(list(zp))})
    rows.append({"predictors": "(sum if independent)", "r2": sum(r["r2"] for r in rows[:len(zp)])})
    return pd.DataFrame(rows)


def _subset_r2(ranks: np.ndarray) -> dict[frozenset[int], float]:
    """R^2 of the rank target (column 0) on every subset of the predictor columns, from one correlation matrix."""
    corr = np.corrcoef(ranks, rowvar=False)
    r2 = {frozenset(): 0.0}
    for size in range(1, corr.shape[0]):
        for subset in combinations(range(1, corr.shape[0]), size):
            idx = list(subset)
            r = corr[0, idx]
            r2[frozenset(subset)] = float(r @ np.linalg.lstsq(corr[np.ix_(idx, idx)], r, rcond=None)[0])
    return r2


def _components(r2: dict[frozenset[int], float], k: int) -> dict[frozenset[int], float]:
    """Variance shared by exactly the predictors in each subset S: sum over T in S of (-1)^(|T|+1) R^2(rest + T)."""
    everyone = range(1, k + 1)
    out = {}
    for size in everyone:
        for s in combinations(everyone, size):
            rest = frozenset(everyone) - frozenset(s)
            out[frozenset(s)] = sum(
                (-1) ** (n + 1) * r2[rest | frozenset(t)]
                for n in range(len(s) + 1) for t in combinations(s, n)
            )
    return out


def _rank_matrix(target: np.ndarray, predictors: dict[str, np.ndarray]) -> np.ndarray:
    mask = pairwise_complete(target, *predictors.values())
    return np.column_stack([rankdata(v[mask]) for v in (target, *predictors.values())])


def commonality(target: np.ndarray, predictors: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Commonality analysis of the rank-transformed `target` on `predictors` (all rank-transformed).

    One row per non-empty subset of predictors (2^k - 1 rows): `kind` is "unique" for a single predictor
    and "shared" otherwise, `value` the share of target variance that exactly that subset accounts for.
    The values sum to the full-model R^2 (also in `attrs["full_r2"]`). Computed on the pairs observed in
    every RDM.
    """
    names = list(predictors)
    r2 = _subset_r2(_rank_matrix(target, predictors))
    rows = [
        {"component": " & ".join(names[i - 1] for i in sorted(s)), "order": len(s),
         "kind": "unique" if len(s) == 1 else "shared", "value": v}
        for s, v in sorted(_components(r2, len(names)).items(), key=lambda kv: (len(kv[0]), sorted(kv[0])))
    ]
    out = pd.DataFrame(rows)
    out.attrs["full_r2"] = r2[frozenset(range(1, len(names) + 1))]
    out.attrs["r2_alone"] = {n: r2[frozenset([i + 1])] for i, n in enumerate(names)}
    return out


def commonality_with_ci(
    target: np.ndarray, predictors: dict[str, np.ndarray], *, n_boot: int = 500, seed: int = 0,
) -> dict:
    """
    Commonality analysis with image-bootstrap 95% CIs (cached) on the unique components, the total shared
    part and the full-model R^2.

    Returns {"components": full component table, "summary": per-predictor unique / shared-with-others /
    R^2 alone (unique with CI), "full_r2", "full_r2_ci", "shared_total", "shared_total_ci"}.
    """
    names = list(predictors)
    k = len(names)

    def stat(*vectors: np.ndarray) -> list[float]:
        r2 = _subset_r2(_rank_matrix(vectors[0], dict(zip(names, vectors[1:]))))
        comps = _components(r2, k)
        unique = [comps[frozenset([i])] for i in range(1, k + 1)]
        full = r2[frozenset(range(1, k + 1))]
        return [*unique, full - sum(unique), full]

    draws = image_bootstrap(
        [target, *predictors.values()], stat, n_boot=n_boot, seed=seed,
        cache_path=cache_path(f"s5_commonality_boot_{'_'.join(names)}.npz"), extra_key=("commonality", names),
    )
    components = commonality(target, predictors)
    alone = components.attrs["r2_alone"]
    unique = components.loc[components["kind"] == "unique"].set_index("component")["value"]
    cis = [percentile_ci(draws[:, i]) for i in range(k)]
    summary = pd.DataFrame({
        "unique": [unique[n] for n in names], "unique_lo": [c[0] for c in cis], "unique_hi": [c[1] for c in cis],
        "shared_with_others": [alone[n] - unique[n] for n in names], "r2_alone": [alone[n] for n in names],
    }, index=pd.Index(names, name="predictor"))
    full = components.attrs["full_r2"]
    return {
        "components": components, "summary": summary, "full_r2": full, "full_r2_ci": percentile_ci(draws[:, k + 1]),
        "shared_total": full - float(unique.sum()), "shared_total_ci": percentile_ci(draws[:, k]),
    }


def plot_unique(result: dict, *, target: str = REFERENCE) -> go.Figure:
    """Bar chart of the unique component per predictor (95% CI bars), coloured by RDM family."""
    fig = go.Figure()
    for name, row in result["summary"].iterrows():
        fig.add_trace(go.Bar(
            x=[plotting.rdm_tick_label(name)], y=[row["unique"]], marker_color=plotting.rdm_style(name).color,
            showlegend=False,
            error_y=dict(type="data", symmetric=False, array=[row["unique_hi"] - row["unique"]],
                         arrayminus=[row["unique"] - row["unique_lo"]], color="#555555"),
            hovertemplate="unique R<sup>2</sup>: %{y:.4f}<extra>" + plotting.rdm_label(name) + "</extra>",
        ))
    lo, hi = result["full_r2_ci"]
    fig.update_layout(
        title=f"Variance in {plotting.rdm_label(target)} that only one predictor explains"
              f"<br><sub>full-model R<sup>2</sup> = {result['full_r2']:.3f} [{lo:.3f}, {hi:.3f}]; "
              f"shared by two or more = {result['shared_total']:.3f}</sub>",
        yaxis=dict(title="unique R<sup>2</sup>", rangemode="tozero"), template=_TEMPLATE, height=460, width=760,
    )
    return fig
