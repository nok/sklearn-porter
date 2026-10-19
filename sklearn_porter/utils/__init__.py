from collections.abc import Callable
from copy import copy
from json import dumps
from logging import ERROR
from math import isinf

import numpy as np
from scipy.sparse import issparse

from sklearn_porter import exceptions as exception

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


def has_infinite_thresholds(tree_data: dict) -> bool:
    """
    Check whether a tree has splits with an infinite threshold.

    Trees fitted on data with missing values (NaN) can split the samples
    with missing values from all other samples with the threshold `inf`.

    Parameters
    ----------
    tree_data : dict
        The model data of a tree with the key `thresholds`.

    Returns
    -------
    Whether the tree has infinite thresholds.
    """
    return any(isinf(t) for t in tree_data['thresholds'])


def with_infinity(
    converter: Callable[[object], str], infinity: str
) -> Callable[[object], str]:
    """
    Extend a number converter by the infinity literal of a language.

    Parameters
    ----------
    converter : Callable[[object], str]
        The number converter.
    infinity : str
        The literal of positive infinity, e.g. `INFINITY` in C.

    Returns
    -------
    The converter which converts infinite values to the literal.
    """

    def convert(value: object) -> str:
        if isinstance(value, float) and isinf(value):
            return infinity if value > 0 else f'-{infinity}'
        return converter(value)

    return convert


def dump_model_data(model_data: object) -> str:
    """
    Dump model data as JSON.

    JSON doesn't support infinite values, so they are written as numbers
    which are too large for a double (`1e999` and `-1e999`). The JSON
    parsers of the supported languages read them as infinite values,
    except Go, which needs a special number type.

    Parameters
    ----------
    model_data : object
        The model data.

    Returns
    -------
    The model data as JSON string.
    """
    out = dumps(model_data, separators=(',', ':'))
    return out.replace('Infinity', '1e999')


def missing_name(language) -> str:
    """
    Get the name of the variable `missing_go_to_left` of trees.

    Parameters
    ----------
    language : Language
        The target programming language.

    Returns
    -------
    The name in the naming convention of the language.
    """
    if language.value.KEY in ('cs', 'go', 'java', 'js'):
        return 'missingGoToLeft'
    return 'missing_go_to_left'


def has_missing_go_to_left(tree_data: dict) -> bool:
    """
    Check whether any split of a tree sends missing values to the left child.

    Parameters
    ----------
    tree_data : dict
        The model data of a tree with the key `missing_go_to_left`, which
        contains only the splits, see `get_tree_data()`.

    Returns
    -------
    Whether the ported tree has to check features for missing values (NaN).
    """
    return any(tree_data['missing_go_to_left'])


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


# The kinds of the supported labels and the ranges of integer labels, which
# have to fit into the `int` type of C and Java:
LABEL_KINDS = {bool: 'int', int: 'int', float: 'double', str: 'str'}
INT_RANGE = (-(2**31), 2**31 - 1)


def get_labels(estimator) -> tuple[list, str]:
    """
    Get the labels of the classes of a fitted classifier and their kind.

    The ported classifiers return the label of the predicted class like
    scikit-learn. All labels must have the same kind, which defines their
    type in the ported code: `int` (also of booleans), `double` or `str`.
    The labels of multi-output classifiers are concatenated like their
    classes (see `get_output_classes()`).

    Parameters
    ----------
    estimator : BaseEstimator
        The fitted classifier.

    Returns
    -------
    The labels and their kind, e.g. `([1, 2, 3], 'int')`.

    Raises
    ------
    NotSupportedYetError
        If the labels have different or unsupported types, are infinite or
        NaN, or don't fit into 32-bit integers.
    """
    classes = estimator.classes_
    if not isinstance(classes, list):
        classes = [classes]
    # `tolist()` converts the NumPy scalars to Python scalars:
    labels = [label for c in classes for label in np.asarray(c).tolist()]
    kinds = {LABEL_KINDS.get(type(label), '') for label in labels}
    if len(kinds) != 1 or '' in kinds:
        types = ', '.join(sorted({type(label).__name__ for label in labels}))
        msg = (
            'The labels of all classes must have the same type '
            f'(int, float or str), but they are of type: {types}.'
        )
        raise exception.NotSupportedYetError(msg)
    kind = kinds.pop()
    if kind == 'int':
        labels = [int(label) for label in labels]
        if not all(INT_RANGE[0] <= label <= INT_RANGE[1] for label in labels):
            msg = 'The labels of the classes must be 32-bit integers.'
            raise exception.NotSupportedYetError(msg)
    elif kind == 'double' and not np.isfinite(labels).all():
        msg = "The labels of the classes can't be infinite or NaN."
        raise exception.NotSupportedYetError(msg)
    return labels, kind


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


def get_tree_data(tree, values, key: str) -> dict:
    """
    Get the model data of a fitted tree in a compact format.

    The arrays of the tree contain only the splits (inner nodes). The
    children of a split refer to another split by its index `i >= 0` or to
    a leaf by `-i - 1`, e.g. `-1` refers to the first leaf. The values of
    all leaves are concatenated, so the values of leaf `i` start at
    `i * n_values`. A tree which consists of a single leaf gets a split
    whose children are both this leaf, so the arrays are never empty.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The fitted tree structure, e.g. `estimator.tree_`.
    values : array-like of shape (n_nodes, n_values)
        The values of all nodes, e.g. the class fractions.
    key : str
        The key of the values of the leaves, e.g. `classes`.

    Returns
    -------
    The model data with the keys `lefts`, `rights`, `thresholds`,
    `indices`, `missing_go_to_left` and `key`.
    """
    is_leaf = tree.children_left == -1
    is_split = ~is_leaf
    leaf_values = np.asarray(values)[is_leaf].ravel().tolist()
    if not is_split.any():
        return {
            'lefts': [-1],
            'rights': [-1],
            'thresholds': [0.0],
            'indices': [0],
            'missing_go_to_left': [0],
            key: leaf_values,
        }
    # The references of all nodes, splits by `i` and leaves by `-i - 1`:
    refs = np.where(is_leaf, -np.cumsum(is_leaf), np.cumsum(is_split) - 1)
    thresholds = np.asarray(get_thresholds(tree))
    return {
        'lefts': refs[tree.children_left[is_split]].tolist(),
        'rights': refs[tree.children_right[is_split]].tolist(),
        'thresholds': thresholds[is_split].tolist(),
        'indices': tree.feature[is_split].tolist(),
        'missing_go_to_left': tree.missing_go_to_left[is_split].tolist(),
        key: leaf_values,
    }


def get_leaf_values(tree_data: dict, key: str) -> list:
    """
    Split the concatenated values of the leaves of a compact tree.

    Parameters
    ----------
    tree_data : dict
        The model data of a tree, see `get_tree_data()`.
    key : str
        The key of the values of the leaves, e.g. `classes`.

    Returns
    -------
    The values of each leaf.
    """
    values = tree_data[key]
    # Each leaf is referenced once, the last one by `-n_leaves`:
    n_leaves = -min(tree_data['lefts'] + tree_data['rights'])
    n_values = len(values) // n_leaves
    return [values[i * n_values : (i + 1) * n_values] for i in range(n_leaves)]
