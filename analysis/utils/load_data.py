"""
Read the participants/trials DataFrames cached as parquet by
analysis.utils.parser.parse_raw_data() (see the parquet-caching scratch-pad script).
Never parses raw data itself -- fails loudly if the cache isn't there.

Usage (from repo root):
    from analysis.utils.load_data import load_data
    data = load_data("analysis/results/parsed_data")
    df_participants = data["participants"]
    df_trials       = data["trials"]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(cache_dir: str | Path, prod_only: bool = False) -> dict[str, pd.DataFrame]:
    """
    Load the cached participants/trials DataFrames from *cache_dir*.

    Raises
    ------
    FileNotFoundError
        If participants.parquet or trials.parquet is missing from *cache_dir*.
    """
    cache_dir = Path(cache_dir)
    participants_path = cache_dir / "participants.parquet"
    trials_path = cache_dir / "trials.parquet"

    missing = [p for p in (participants_path, trials_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cached parquet file(s) not found: {[str(p) for p in missing]}. "
            "Run the parquet-caching script first."
        )

    df_participants = pd.read_parquet(participants_path)
    df_trials = pd.read_parquet(trials_path)

    if prod_only:
        df_participants = df_participants[df_participants["cohort"] == "production"]
        df_trials = df_trials[df_trials["participant_id"].isin(df_participants["participant_id"])]

    return {"participants": df_participants, "trials": df_trials}
