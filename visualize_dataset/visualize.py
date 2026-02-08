from typing import Dict

import PyQt5
import ete3

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_tree(tree: str | ete3.Tree):
    if isinstance(tree, str):
        tree = ete3.Tree(tree, format=8)
    ts = ete3.TreeStyle()
    ts.show_leaf_name = True
    ts.branch_vertical_margin = 10
    ts.rotation = 90    # rotate tree for better visibility
    tree.show(tree_style=ts)


def subcategory_size_distribution(subcategory_counts: Dict[tuple, int]) -> go.Figure:
    """ Creates a multi-plot figure showing histograms of instance-distributions across different top-level categories. """
    n_boxes = max(subcategory_counts.values()) + 1
    subplot_titles = {
        1: {"All Categories": "#969696"},
        2: {"Animate": "#238b45", "Inanimate": "#ef3b2c"},
        3: {"Animate/Animal": "#41ab5d", "Animate/Human": "#006d2c",
            "Inanimate/Handmade": "#a50f15", "Inanimate/Natural": "#fb6a4a"},
    }
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
    )
    return fig

