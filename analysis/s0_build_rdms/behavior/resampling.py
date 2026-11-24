"""
Delete-one participant jackknife of a combined behavioral RDM: the statistic of interest is recomputed with
each participant left out in turn, and its standard error comes from the spread of those values
(analysis.utils.stats.jackknife_se / jackknife_ci).

Why not a participant bootstrap: on these sparse RDMs a statistic of the combined RDM grows with the number
of participants (extra participants mostly add newly covered pairs), and a with-replacement resample has only
about 63% distinct participants, so every bootstrap draw is attenuated and the percentile CI sits below the
estimate. A leave-one-out RDM keeps all but one participant, so the CI is centred on the estimate. The
jackknife's own bias estimate stays large for the same reason; it says how far a 155-participant value is
from an infinitely large sample's, and is reported, never subtracted.

Usage (from repo root):
    from analysis.s0_build_rdms.behavior.resampling import participant_jackknife
    rows = participant_jackknife(trials, n_conds, lambda rdm: [{"rho": spearman_rho(...)}])
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analysis.s0_build_rdms.behavior.spam_ratings import _combine

TrialsByParticipant = dict[str, list[tuple[np.ndarray, np.ndarray]]]
Score = Callable[[np.ndarray], list[dict]]


def _leave_out(pids: list[str], trials: TrialsByParticipant, n_conds: int, aggregation: str, score: Score) -> list[dict]:
    rows = []
    for pid in pids:
        rdm = _combine([t for other, ts in trials.items() if other != pid for t in ts], n_conds, aggregation)[0]
        rows += [{"left_out": pid, **row} for row in score(rdm)]
    return rows


def participant_jackknife(
    trials: TrialsByParticipant, n_conds: int, score: Score, *, aggregation: str = "simple", n_jobs: int = 6,
) -> pd.DataFrame:
    """`score` applied to the combined RDM of every leave-one-participant-out sample: one row per (left-out
    participant, row returned by `score`), with the participant id in `left_out`. Participants are spread
    over `n_jobs` worker processes; the result does not depend on `n_jobs`."""
    ids = list(trials)
    chunks = [ids[i::n_jobs] for i in range(n_jobs)] if n_jobs > 1 else [ids]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_leave_out)(chunk, trials, n_conds, aggregation, score) for chunk in chunks if chunk
    )
    order = {pid: i for i, pid in enumerate(ids)}
    frame = pd.DataFrame([row for part in parts for row in part])
    return frame.sort_values("left_out", key=lambda s: s.map(order), kind="stable", ignore_index=True)
