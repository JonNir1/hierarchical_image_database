"""Power for RQ2a(ii), measured from simulated cohorts rather than approximated.

RQ2a(ii) asks whether ``rho(pre, post)`` sits **below** the within-condition ceiling, i.e. whether
the two SHINE cohorts differ by more than subject-sampling noise. Its null is the pre-registration's
cross-subject shuffle: pool everyone, split at random ignoring condition, correlate. Two ingredients:

* the **null** - two cohorts of N drawn from the *same* ground truth. This is exactly what
  ``pipeline.embedding_stability_draws`` produces, so the main run supplies it for free.
* the **alternative** - a cohort of N on the pre-SHINE ground truth against a cohort of N on a
  perturbed one, from ``pipeline.cross_gt_draws``.

Power is then the share of alternative draws falling below the null's ``alpha`` quantile. No normal
approximation, no classical-attenuation formula, and no assumption about how the sampling spread
scales with N - all three of which an analytic version needs, and the last of which is the one that
cannot be justified from the data.

**What this does not model.** The perturbation is isotropic (see
``empirical.gt_perturbation``), the two cohorts are simulated from ground truths that differ only by
that perturbation, and both are drawn from the same participant population. A real SHINE effect that
also changed how participants *behave* - rather than only what the images look like - is outside it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_ALPHA = 0.05
CELL_FIELDS = ("num_subjects", "screening_min_reliability")


def _cell_key(frame: pd.DataFrame, cell_fields: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in cell_fields if c not in frame.columns]
    if missing:
        raise ValueError(f"draws are missing the cell columns {missing}; got {list(frame.columns)}")
    return frame


def power_table(null_draws: pd.DataFrame, alt_draws: pd.DataFrame, *,
                target_rho: float, alpha: float = DEFAULT_ALPHA,
                cell_fields: Sequence[str] = CELL_FIELDS) -> pd.DataFrame:
    """Power per cell for one perturbation strength.

    ``null_draws`` and ``alt_draws`` each carry a ``spearman`` column plus the cell columns.
    The critical value is the null's ``alpha`` quantile **within each cell**, since the ceiling and
    its spread both depend on N and on the screening threshold.

    Also returns ``ceiling`` (the null's mean, i.e. the within-condition reliability r(N)) and
    ``observed`` (the alternative's mean, i.e. what rho(pre, post) would be), because the power
    number alone is not interpretable without the two correlations it came from.
    """
    _cell_key(null_draws, cell_fields)
    _cell_key(alt_draws, cell_fields)
    keys = list(cell_fields)
    rows = []
    alt_by_cell = {k: g for k, g in alt_draws.groupby(keys, sort=False)}
    for key, null_grp in null_draws.groupby(keys, sort=False):
        alt_grp = alt_by_cell.get(key)
        if alt_grp is None or null_grp.empty:
            continue
        null_vals = null_grp["spearman"].to_numpy(dtype=float)
        alt_vals = alt_grp["spearman"].to_numpy(dtype=float)
        critical = float(np.quantile(null_vals, alpha))
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(keys, key_tuple)),
            "target_rho": float(target_rho),
            "n_null_draws": int(null_vals.size), "n_alt_draws": int(alt_vals.size),
            "ceiling": float(null_vals.mean()),
            "ceiling_sd": float(null_vals.std(ddof=1)) if null_vals.size > 1 else np.nan,
            "observed": float(alt_vals.mean()),
            "drop_below_ceiling": float(null_vals.mean() - alt_vals.mean()),
            "critical_value": critical,
            "power": float(np.mean(alt_vals < critical)),
        })
    return pd.DataFrame(rows)


def power_curve(null_draws: pd.DataFrame, alt_draws_by_rho: dict, *,
                alpha: float = DEFAULT_ALPHA,
                cell_fields: Sequence[str] = CELL_FIELDS) -> pd.DataFrame:
    """:func:`power_table` over several perturbation strengths, stacked.

    ``alt_draws_by_rho`` maps ``target_rho -> cross_gt_draws frame``. The null is shared: it does not
    depend on the perturbation, which is the whole reason the main run supplies it at no extra cost.
    """
    frames = [power_table(null_draws, alt, target_rho=rho, alpha=alpha, cell_fields=cell_fields)
              for rho, alt in sorted(alt_draws_by_rho.items(), reverse=True)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def minimum_detectable_effect(curve: pd.DataFrame, *, power_target: float = 0.80,
                              cell_fields: Sequence[str] = CELL_FIELDS) -> pd.DataFrame:
    """The largest ``target_rho`` each cell can detect at ``power_target``, by interpolation.

    Power rises as ``target_rho`` falls (a bigger SHINE effect is easier to see), so the curve is
    interpolated on a reversed axis. Cells whose power never reaches the target within the simulated
    range return NaN rather than an extrapolated value: the honest answer is "not within the range
    we simulated", and an extrapolated rho would read as a measurement.
    """
    keys = list(cell_fields)
    rows = []
    for key, grp in curve.groupby(keys, sort=False):
        grp = grp.sort_values("target_rho")           # ascending rho == descending power
        rho, power = grp["target_rho"].to_numpy(float), grp["power"].to_numpy(float)
        key_tuple = key if isinstance(key, tuple) else (key,)
        if power.max() < power_target:
            mde = np.nan
        elif power.min() >= power_target:
            mde = float(rho.max())                    # even the smallest simulated effect is caught
        else:
            # np.interp needs an increasing x; power decreases in rho, so reverse both.
            mde = float(np.interp(power_target, power[::-1], rho[::-1]))
        rows.append({**dict(zip(keys, key_tuple)), "power_target": power_target,
                     "min_detectable_rho": mde,
                     "min_detectable_effect_pct": (np.nan if np.isnan(mde) else 100 * (1 - mde))})
    return pd.DataFrame(rows)
