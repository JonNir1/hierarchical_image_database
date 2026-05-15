# CLAUDE.md — visualize_dataset

Tools for parsing the hierarchical image dataset and visualizing its category structure.

## Architecture

`parse.py` ingests a directory tree where path structure encodes the category hierarchy:
`<dataset_root>/<cat1>/<cat2>/.../<catN>/<name><instance_number>.png`

For this project the conventional `<dataset_root>` is `<repo>/images/pre_shine/` or
`<repo>/images/post_shine/` (one canonical location per SHINE variant). Pass the
desired variant root as `dataset_path` to `as_frame()`.

- `as_frame(dataset_path)` → DataFrame with columns `cat1..catN`, `instance`, and image metadata (height, width, channels, alpha).
- `as_counts(dataset)` → dict mapping each sub-category tuple to its instance count.
- `as_tree(dataset)` → ete3 Tree (or Newick string) with instance counts at each node.

The bottom-most category name is extracted from the filename by stripping trailing digits; the instance number is the trailing digit sequence (e.g. `dog01.png` → category `dog`, instance `1`).

`visualize.py` provides Plotly (`subcategory_size_distribution`) and ete3/PyQt5 (`show_tree`) visualizations over the parsed DataFrame or tree.
