# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.

How do the category hierarchies of the Auerbach-Asch 2023 and PiCS 2025 datasets compare, and how large are
their subcategories? **This notebook was not re-run.** It needs `ete3`, `PyQt5` and the two external dataset
folders, none of which are available here. Its stored outputs predate the fixes in commit `a6f971bc` (notably
the PiCS category-count cell, below), so treat them as historical.

- **Caveat, stored outputs are old:** the stored figures were drawn with the previous hard-coded colors and no
  layout size. The restyled figure code (`visualize.py`) and the new section for `images/pre_shine` have not been executed in the notebook.
- **Optional dependencies:** `ete3` is imported only by `as_tree` and `show_tree`, and `PyQt5` only by `show_tree`; neither is in `requirements.txt`.

## How many categories does each dataset have?

Notebook: [`dataset_visualizations.ipynb`](dataset_visualizations.ipynb) (commit `a6f971bc`)

- **Auerbach-Asch folder:** 301 categories (distinct category paths down to the leaf).
- **PiCS `hierarchical` folder:** the stored output says 264, but that number is unreliable. The cell computed it from the Auerbach-Asch frame (`aa_df`) instead of the PiCS frame, so it counts Auerbach-Asch rows over PiCS's category columns. The cell is fixed; the count needs a re-run with the PiCS folder.
- **This project's `images/pre_shine`** (ad hoc check, not stored in the notebook): 725 images and 301 categories, equal to the Auerbach-Asch count, the largest category holding 31 images.

## How are subcategory sizes distributed?

Notebook: [`dataset_visualizations.ipynb`](dataset_visualizations.ipynb) (commit `a6f971bc`)

- **Figures only:** the histograms of instances per category (all categories, animate vs inanimate, and the four sub-domains) were saved as figures in the notebook and printed no numbers, so there is nothing to quote here.
- **Caveat:** `parse.as_counts` compares every row against every other row (quadratic); fine at 725 images, slow on large datasets.
