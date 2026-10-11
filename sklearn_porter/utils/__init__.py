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
