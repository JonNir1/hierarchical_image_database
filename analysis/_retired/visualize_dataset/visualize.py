"""
Figures for the dataset category hierarchy: an interactive ete3 tree viewer and the subcategory-size histograms.

Usage (from repo root):
    from analysis._retired.visualize_dataset.visualize import show_tree, subcategory_size_distribution

``ete3`` and ``PyQt5`` are only needed by ``show_tree`` and are imported lazily.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.utils.plotting import role_palette

if TYPE_CHECKING:
    import ete3

_TEMPLATE = "plotly_white"
_ALL_COLOR = "#969696"  # neutral grey: the pooled panel belongs to no category


def show_tree(tree: str | ete3.Tree) -> None:
    import ete3
    import PyQt5  # noqa: F401  (ete3's tree viewer needs a Qt backend)

    if isinstance(tree, str):
        tree = ete3.Tree(tree, format=8)
    ts = ete3.TreeStyle()
    ts.show_leaf_name = True
    ts.branch_vertical_margin = 10
    ts.rotation = 90    # rotate tree for better visibility
    tree.show(tree_style=ts)


def _panel_colors() -> dict[int, dict[str, str]]:
    """Hue per top-level domain (animate/inanimate); sub-domains reuse their parent's hue, the second at lower alpha."""
    domains = role_palette(["animate", "inanimate"])
    return {
        1: {"All Categories": _ALL_COLOR},
        2: {"Animate": domains["animate"]["main"], "Inanimate": domains["inanimate"]["main"]},
        3: {"Animate/Animal": domains["animate"]["main"], "Animate/Human": domains["animate"]["dot_err"],
            "Inanimate/Handmade": domains["inanimate"]["main"], "Inanimate/Natural": domains["inanimate"]["dot_err"]},
    }


def subcategory_size_distribution(subcategory_counts: Dict[tuple, int]) -> go.Figure:
    """ Creates a multi-plot figure showing histograms of instance-distributions across different top-level categories. """
    n_boxes = max(subcategory_counts.values()) + 1
    subplot_titles = _panel_colors()
    fig = make_subplots(
        rows=3, cols=4,
        specs=[
            [{"colspan": 4}, None, None, None],
            [{"colspan": 2}, None, {"colspan": 2}, None],
            [{}, {}, {}, {}],
        ],
        x_title="Category Size", y_title="Number of Categories",
    )
    for r in subplot_titles.keys():
        for c, (category, color) in enumerate(subplot_titles[r].items()):
            row = r
            col = 1 if r == 1 else c * 2 + 1 if r == 2 else c + 1
            if category == "All Categories":
                filtered_counts = subcategory_counts
            else:
                category_tuple = tuple(category.lower().split("/"))
                filtered_counts = {k: v for k, v in subcategory_counts.items() if k[:len(category_tuple)] == category_tuple}
            fig.add_trace(
                row=row, col=col,
                trace=go.Histogram(
                    x=list(filtered_counts.values()),
                    name=category,
                    marker=dict(color=color),
                    nbinsx=n_boxes,
                )
            )
            fig.add_annotation(
                row=row, col=col,
                x=0.5, xanchor="center", xref="x domain",
                y=1.0, yanchor="top", yref="y domain",
                text=f"<b>{category}</b>",
                showarrow=False,
                font=dict(size=16, color=color),
            )
    fig.update_layout(
        title=dict(text="Instance Distribution - Top-Level Categories"),
        legend=dict(visible=False),
        template=_TEMPLATE,
        height=800, width=1000,
    )
    return fig
