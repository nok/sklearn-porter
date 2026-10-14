from copy import copy
from logging import ERROR

import numpy as np
from scipy.sparse import issparse

options = {'logging.level': ERROR}


def densify(estimator):
    """
    Convert the sparse matrices of a fitted estimator to dense arrays.

    Estimators which were fitted on sparse data (e.g. `SVC` or
    `KNeighborsClassifier`) or converted with `sparsify()` (e.g.
    `LogisticRegression`) store sparse matrices, but the ported estimators
    need dense arrays. The passed estimator remains unchanged.

    Parameters
    ----------
    estimator : BaseEstimator
        The fitted estimator.

    Returns
    -------
    A shallow copy of the estimator with dense arrays instead of sparse
    matrices, or the estimator itself if it doesn't store sparse matrices.
    """
    sparse = {k: v for k, v in vars(estimator).items() if issparse(v)}
    if not sparse:
        return estimator
    estimator = copy(estimator)
    for key, value in sparse.items():
        setattr(estimator, key, value.toarray())
    return estimator


def is_multilabel(estimator) -> bool:
    """
    Check whether a classifier was fitted on a label indicator matrix.

    Multilabel classifiers (e.g. `RidgeClassifier` or `MLPClassifier`) don't
    predict a single class per sample, but one binary value per label.
    scikit-learn checks the type of the target in the same way.

    Parameters
    ----------
    estimator : BaseEstimator
        The fitted estimator.

    Returns
    -------
    Whether the estimator is a multilabel classifier.
    """
    binarizer = getattr(estimator, '_label_binarizer', None)
    y_type = getattr(binarizer, 'y_type_', '')
    return y_type.startswith('multilabel')


def get_output_classes(estimator) -> list:
    """
    Get the number of classes per output of a fitted classifier.

    Multi-output classifiers (e.g. a DecisionTreeClassifier fitted on a target
    with several columns) store the classes of each output in a list. The
    ported multi-output classifiers concatenate the class counts (or
    probabilities) of all outputs, so this list defines their segments.

    Parameters
    ----------
    estimator : BaseEstimator
        The fitted classifier.

    Returns
    -------
    The number of classes per output, e.g. `[3]` for a single output.
    """
    classes = estimator.classes_
    if isinstance(classes, list):
        return [len(c) for c in classes]
    return [len(classes)]


def get_class_fractions(tree) -> list:
    """
    Get the (weighted) fractions of the classes per node of a fitted tree.

    Since scikit-learn v1.4 `tree_.value` contains the class fractions of
    classifiers, which scikit-learn uses to predict the probabilities. They
    must not be rounded to class counts, because the samples can be weighted
    (e.g. by `sample_weight` or `class_weight`). The fractions of multiple
    outputs are concatenated.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The fitted tree structure, e.g. `estimator.tree_`.

    Returns
    -------
    The class fractions of shape (n_nodes, sum of the classes of all outputs).
    """
    # The values are padded to the maximum number of classes of all outputs:
    return np.hstack(
        [tree.value[:, i, :n] for i, n in enumerate(tree.n_classes)]
    ).tolist()


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
