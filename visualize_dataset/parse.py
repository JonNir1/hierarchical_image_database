import os
from glob import glob
from typing import Dict, Set
from collections import defaultdict

import pandas as pd
import cv2
import ete3
from tqdm import tqdm


def as_frame(dataset_path: str, extensions: str | list[str] = "png") -> pd.DataFrame:
    """
    Iterates over all files in dataset_path with the given extensions, extracts metadata and category/instance information,
    and returns a DataFrame summarizing the dataset, with one row per image and the following columns:
    - cat1, cat2, ..., catN: category levels extracted from the directory structure
    - instance: instance of the specific bottom-level category, extracted from the filename (e.g. "dog01.png" -> instance 1)
    - height: image height in pixels
    - width: image width in pixels
    - n_channels: number of color channels (e.g. 3 for RGB, 4 for RGBA)
    - has_alpha: boolean indicating if the image has an alpha channel
    - extension: file extension (e.g. ".png")
    - path: relative path to the image from dataset_path
    """
    assert os.path.isdir(dataset_path), f"{dataset_path} is not a directory"
    if isinstance(extensions, str):
        extensions = [extensions]
    assert extensions, f"extensions must be a string or list of strings"
    image_paths = []
    for ext in extensions:
        image_paths = glob(os.path.join(dataset_path, f"**{os.sep}*.{ext}"), recursive=True)
    records = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            record = _extract_image_metadata(image_path, base_path=dataset_path)
            records.append(record)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    df = pd.DataFrame.from_records(records).fillna(pd.NA)
    df = df[
        sorted([col for col in df.columns if col.startswith("cat")], key=lambda col: int(col.replace("cat", ""))) +
        ["instance", "height", "width", "n_channels", "has_alpha", "extension", "path"]
    ]
    return df


def as_counts(dataset: str | pd.DataFrame) -> Dict[tuple, int]:
    """ Count instances for each unique sub-category in the dataset. """
    dataset = dataset if isinstance(dataset, pd.DataFrame) else as_frame(dataset)
    cleaned = dataset.copy().fillna(-1)     # use -1 as the fill value for missing categories
    category_columns = sorted(
        [col for col in cleaned.columns if col.startswith('cat')],
        key=lambda col: int(col.replace('cat', ''))
    )
    sub_df = cleaned[category_columns]
    counts = defaultdict(int)
    for _, row in sub_df.iterrows():
        sub_category = row[category_columns].tolist()
        sub_category_cleaned = tuple([cat for cat in sub_category if cat != -1])
        if not sub_category_cleaned:
            continue  # skip empty categories
        if tuple(sub_category_cleaned) in counts:
            continue  # already counted this sub-category
        n_instances = sub_df[sub_df.eq(row[category_columns]).all(axis=1)].shape[0]
        counts[sub_category_cleaned] = n_instances
    return counts


def as_tree(dataset: str | pd.DataFrame, as_newick: bool = False) -> str | ete3.Tree:
    """ Parse the dataset as a tree representation, either in Newick (str) format or as an ete3.Tree object. """
    leaf_counts = as_counts(dataset)
    tree, counts = _build_tree(leaf_counts, update_internal_counts=True)

    def newick_subtree(node):
        """ Converts the subtree rooted at `node` to Newick format. """
        name = node[-1] if node else "ROOT"
        kids = sorted(tree.get(node, []), key=lambda t: t[-1])
        label = f"'{name}-{counts[node]}'"
        return f"({','.join(newick_subtree(k) for k in kids)}){label}" if kids else label
    ete3_tree = ete3.Tree(newick_subtree(()) + ";", format=8)
    if as_newick:
        return ete3_tree.write(format=8)
    return ete3_tree


def _extract_image_metadata(image_path: str, base_path: str = ""):
    """
    Extract the image's category (from the path structure), instance (from the filename), and metadata such as height,
    width, channels, has_alpha, extension, and relative path.
    """
    assert image_path.startswith(base_path), f"{image_path} does not start with {base_path}"
    rel_path = os.path.relpath(image_path, base_path)
    parts = rel_path.split(os.sep)
    name, extension = os.path.splitext(parts[-1])
    name = name.strip()
    assert name[-1].isdigit(), f"Image {rel_path} does not end with a digit"
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise TypeError(f"Could not read image {rel_path}")
    assert img.ndim == 3, f"Image {rel_path} does not have 3 dimensions (height, width, channels)"
    height, width, n_channels = img.shape
    has_alpha = n_channels == 4
    metadata = {
        "height": height,
        "width": width,
        "n_channels": n_channels,
        "has_alpha": has_alpha,
        "extension": extension,
        "path": rel_path,
    }
    categories = {f"cat{i+1}": cat.strip() for i, cat in enumerate(parts[:-1])}
    categories[f"cat{len(categories)+1}"] = "".join([ch for ch in name if not ch.isdigit()]).replace(" ", "_")
    categories["instance"] = int("".join([ch for ch in name if ch.isdigit()]))
    return {**categories, **metadata, }


def _build_tree(counts: dict, update_internal_counts: bool = True) -> (Dict[tuple, Set[tuple]], Dict[tuple, int]):
    """
    Build a tree representation of the category hierarchy from counts.
    If `update_internal_counts` is True, the input `counts` is updated to include counts for internal nodes.
    """
    tree = defaultdict(set)
    if update_internal_counts:
        new_counts = counts.copy()
    for sub_category in counts.keys():
        for i in range(1, len(sub_category)):
            parent, child = sub_category[:i], sub_category[:i + 1]
            tree[parent].add(child)     # insure parent exists in tree
            if update_internal_counts:
                new_counts[parent] += counts[sub_category]     # insure parent exists in new_counts
    top_categories = [cat for cat in tree.keys() if len(cat) == 1]
    tree[()] = set(top_categories)  # insure root node exists in tree
    if update_internal_counts:
        new_counts[()] = sum([new_counts[cat] for cat in top_categories])  # total count of top-level categories
        return tree, new_counts
    return tree, counts
