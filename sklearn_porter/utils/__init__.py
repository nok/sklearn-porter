from logging import ERROR

import numpy as np

options = {'logging.level': ERROR}


def get_class_counts(tree) -> list:
    """
    Get the weighted number of samples per class and node of a fitted tree.

    Since scikit-learn v1.4 `tree_.value` contains the class fractions of
    classifiers instead of the absolute class counts, but all templates
    expect integer counts.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The fitted tree structure, e.g. `estimator.tree_`.

    Returns
    -------
    The class counts of shape (n_nodes, n_classes).
    """
    fractions = tree.value[:, 0, :]
    counts = fractions * tree.weighted_n_node_samples[:, None]
    return np.rint(counts).astype(int).tolist()


def get_thresholds(tree) -> list:
    """
    Get the thresholds of a fitted tree for features in double precision.

    scikit-learn converts the features to single precision (float32) before
    it compares them with the thresholds (`float32(x) <= threshold`). The
    ported estimators compare the features in double precision, which leads
    to different decisions if a feature is (almost) equal to a threshold,
    e.g. `1.9 <= float32(1.9)` is false, but `float32(1.9) <= float32(1.9)`
    is true. Therefore each threshold is replaced by the largest double
    `t` for which `x <= t` and `float32(x) <= threshold` are equivalent.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The fitted tree structure, e.g. `estimator.tree_`.

    Returns
    -------
    The thresholds of shape (n_nodes,); leaves keep the value -2.
    """
    is_leaf = tree.children_left == -1
    return [
        float(threshold) if leaf else _to_double_threshold(threshold)
        for threshold, leaf in zip(tree.threshold, is_leaf, strict=True)
    ]


def _to_double_threshold(threshold: float) -> float:
    # The largest float32 value `v` with `v <= threshold`:
    v = np.float32(threshold)
    if v > threshold:
        v = np.nextafter(v, np.float32(-np.inf))
    # All doubles below the midpoint of `v` and the next float32 value
    # are rounded down to `v` or below. The midpoint itself is rounded
    # to the value with the even mantissa ("round half to even"):
    upper = np.nextafter(v, np.float32(np.inf))
    midpoint = (float(v) + float(upper)) / 2  # exact in double precision
    if np.float32(midpoint) == v:
        return midpoint
    return float(np.nextafter(midpoint, -np.inf))
