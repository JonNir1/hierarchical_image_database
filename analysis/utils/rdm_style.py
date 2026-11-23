"""
One design language for every figure that shows RDMs side by side: an RDM's name fixes how it is drawn.

Three independent properties of an RDM name map to three independent visual channels, so a reader can
decode any marker or line without a legend lookup:

  family          (what is measured)         -> hue
      km = the curated Kiani-Mur tree (the reference: near-black, dashed), wn = WordNet, clip = CLIP,
      sens = pixels, behav = SpAM behavior. The four non-reference families take the four CVD-validated
      hues of analysis/utils/plotting.py, so no unvalidated hue is introduced.
  variant         (which stimuli / cohort)    -> marker symbol, and shade within the family hue
      pre = circle (full hue), post = diamond (lighter), joint = star (full hue), none (sem_*) = square.
  representation  (raw RDM or MDS embedding)  -> marker fill (and line dash)
      raw = filled, mds = the open version of the same symbol; a low-d embedding is a lighter shade
      than a high-d one (dash-dot vs dotted lines).

Names follow the analysis/results/rdms/ stems: sem_km, sem_wn, clip_pre, sens_post, behav_joint,
behav_joint_mds_low / behav_joint_mds_high, and optionally a dimensionality (behav_joint_mds9).

Usage (from repo root):
    from analysis.utils import rdm_style
    fig.add_trace(go.Scatter(x=x, y=y, **rdm_style.scatter_kwargs("clip_pre")))
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from analysis.utils.plotting import _VALIDATED_HUES

REFERENCE_COLOR = "#222222"
FAMILY_HUE: dict[str, str] = {
    "km": REFERENCE_COLOR,
    "wn": _VALIDATED_HUES[2],     # green
    "clip": _VALIDATED_HUES[0],   # blue
    "sens": _VALIDATED_HUES[1],   # orange
    "behav": _VALIDATED_HUES[3],  # purple
}
FAMILY_LABEL = {"km": "KM tree", "wn": "WordNet", "clip": "CLIP", "sens": "pixels", "behav": "behavior"}
VARIANT_SYMBOL = {None: "square", "pre": "circle", "post": "diamond", "joint": "star"}
VARIANT_LIGHTEN = {None: 0.0, "pre": 0.0, "post": 0.45, "joint": 0.0}

_FAMILY_ORDER = list(FAMILY_HUE)
_VARIANT_ORDER = [None, "pre", "post", "joint"]
_NAME = re.compile(r"^(?P<stem>sem_km|sem_wn|clip|sens|behav)(?:_(?P<variant>pre|post|joint))?"
                   r"(?:_(?P<repr>mds)(?P<ndim>\d+)?(?:_(?P<tier>low|high))?)?$")
_STEM_FAMILY = {"sem_km": "km", "sem_wn": "wn", "clip": "clip", "sens": "sens", "behav": "behav"}


@dataclass(frozen=True)
class RdmStyle:
    family: str
    variant: str | None
    representation: str
    ndim: int | None
    tier: str | None
    color: str
    symbol: str
    dash: str
    label: str


def _lighten(hex_color: str, amount: float) -> str:
    """Mix `hex_color` toward white by `amount` in [0, 1]."""
    rgb = [int(hex_color.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4)]
    mixed = [round(c + (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def parse(name: str) -> tuple[str, str | None, str, int | None]:
    """(family, variant, representation, ndim) of an RDM name; ValueError for an unknown name."""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"not an RDM name this style knows: {name!r}")
    ndim = int(m["ndim"]) if m["ndim"] else None
    return _STEM_FAMILY[m["stem"]], m["variant"], "mds" if m["repr"] else "raw", ndim


def _tier(name: str) -> str | None:
    m = _NAME.match(name)
    return m["tier"] if m else None


def label(name: str) -> str:
    """Human-readable label, e.g. 'CLIP pre', 'behavior joint, MDS d=9'."""
    family, variant, representation, ndim = parse(name)
    text = " ".join(filter(None, [FAMILY_LABEL[family], variant]))
    if representation == "mds":
        tier = _tier(name)
        text += ", MDS" + (f" d={ndim}" if ndim else "") + (f" {tier}-d" if tier else "")
    return text


def style(name: str, *, lighten: float = 0.0) -> RdmStyle:
    """Full visual specification of RDM `name`. `lighten` adds extra whitening (e.g. to grade a sweep)."""
    family, variant, representation, ndim = parse(name)
    tier = _tier(name)
    tier_lighten = 0.4 if tier == "low" else 0.0
    color = _lighten(FAMILY_HUE[family], min(1.0, VARIANT_LIGHTEN[variant] + tier_lighten + lighten))
    symbol = VARIANT_SYMBOL[variant] + ("-open" if representation == "mds" else "")
    dash = "dash" if family == "km" else ("solid" if representation == "raw" else ("dashdot" if tier == "low" else "dot"))
    return RdmStyle(family, variant, representation, ndim, tier, color, symbol, dash, label(name))


def marker(name: str, *, size: int = 10, lighten: float = 0.0) -> dict:
    s = style(name, lighten=lighten)
    return {"color": s.color, "symbol": s.symbol, "size": size, "line": {"color": s.color, "width": 1.5}}


def line(name: str, *, width: float = 2.5, lighten: float = 0.0) -> dict:
    s = style(name, lighten=lighten)
    return {"color": s.color, "dash": s.dash, "width": width}


def scatter_kwargs(name: str, *, size: int = 10, width: float = 2.5, lighten: float = 0.0) -> dict:
    """Keyword arguments for go.Scatter: name, marker, line and a per-family legend group."""
    s = style(name, lighten=lighten)
    return {
        "name": s.label, "legendgroup": s.family,
        "marker": marker(name, size=size, lighten=lighten), "line": line(name, width=width, lighten=lighten),
    }


def legend_order(names: list[str]) -> list[str]:
    """`names` sorted by family (reference first), then variant, then raw before MDS, then ndim."""
    def key(name: str) -> tuple:
        family, variant, representation, ndim = parse(name)
        tier = {None: 0, "low": 1, "high": 2}[_tier(name)]
        return (_FAMILY_ORDER.index(family), _VARIANT_ORDER.index(variant), representation != "raw", tier, ndim or 0)
    return sorted(names, key=key)


def tick_label(name: str) -> str:
    """HTML tick text in the RDM's colour, for categorical axes (heatmaps, bar charts)."""
    s = style(name)
    return f'<span style="color:{s.color}">{s.label}</span>'
