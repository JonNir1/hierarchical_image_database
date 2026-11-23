"""Tests for analysis.utils.rdm_style: the name -> visual-channel mapping every s2 figure relies on."""
from __future__ import annotations

from analysis.utils import rdm_style

_NAMES = ["sem_km", "sem_wn", "clip_pre", "clip_post", "sens_pre", "sens_post", "behav_joint", "behav_joint_mds_low", "behav_joint_mds_high"]


def test_channels_are_independent_and_unambiguous():
    styles = {n: rdm_style.style(n) for n in _NAMES}
    assert len({(s.color, s.symbol) for s in styles.values()}) == len(_NAMES)
    for family in ("clip", "sens"):
        pre, post = styles[f"{family}_pre"], styles[f"{family}_post"]
        assert pre.family == post.family and pre.color != post.color
        assert (pre.symbol, post.symbol) == ("circle", "diamond")
    raw, low, high = styles["behav_joint"], styles["behav_joint_mds_low"], styles["behav_joint_mds_high"]
    assert raw.color == high.color and high.symbol == low.symbol == raw.symbol + "-open"
    assert low.color != high.color
