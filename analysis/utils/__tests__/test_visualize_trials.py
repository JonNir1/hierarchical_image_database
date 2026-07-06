import warnings

import pandas as pd
import pytest

from analysis.utils.visualize_trials import visualize_trials


def _make_trials_df(n_trials: int, repeats: dict[int, int]) -> pd.DataFrame:
    """
    Build a minimal trials DataFrame with *n_trials* trials (numbered
    1..n_trials, presented in that order). *repeats* maps
    {repeat_trial_number: original_trial_number}.

    final_locations is left empty so render_trial() returns a blank canvas
    without touching any image files.
    """
    rows = []
    for trial_number in range(1, n_trials + 1):
        orig_num = repeats.get(trial_number)
        rows.append({
            "trial_number": trial_number,
            "is_trial_repeat": orig_num is not None,
            "repeat_of_trial_number": float(orig_num) if orig_num is not None else float("nan"),
            "final_locations": "",
            "participant_id": "test_subject",
        })
    return pd.DataFrame(rows)


def _subplot_titles(fig) -> list[str]:
    return [a.text for a in fig.layout.annotations]


def test_only_repeats_true_selects_pairs_in_presentation_order():
    # trial 5 repeats trial 2; trial 9 repeats trial 4.
    df = _make_trials_df(n_trials=10, repeats={5: 2, 9: 4})

    fig = visualize_trials(df, only_repeats=True)

    assert _subplot_titles(fig) == ["Trial 2", "Trial 5", "Trial 4", "Trial 9"]


def test_only_repeats_false_first_subplots_match_only_repeats_true():
    df = _make_trials_df(n_trials=10, repeats={5: 2, 9: 4})

    fig_pairs = visualize_trials(df, only_repeats=True)
    fig_all = visualize_trials(df, only_repeats=False)

    pair_titles = _subplot_titles(fig_pairs)
    all_titles = _subplot_titles(fig_all)

    assert all_titles[: len(pair_titles)] == pair_titles
    # remaining trials (not part of any pair) follow, in presentation order
    assert all_titles[len(pair_titles):] == [
        "Trial 1", "Trial 3", "Trial 6", "Trial 7", "Trial 8", "Trial 10",
    ]


def test_only_repeats_true_with_no_repeats_warns_and_falls_back():
    df = _make_trials_df(n_trials=5, repeats={})

    with pytest.warns(UserWarning, match="no test-retest repeat trials"):
        fig_fallback = visualize_trials(df, only_repeats=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig_all = visualize_trials(df, only_repeats=False)

    assert _subplot_titles(fig_fallback) == _subplot_titles(fig_all)
    assert _subplot_titles(fig_all) == [f"Trial {i}" for i in range(1, 6)]
