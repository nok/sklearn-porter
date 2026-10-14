import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_multilabel_classification,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter.utils import (
    densify,
    get_class_fractions,
    get_output_classes,
    get_thresholds,
    has_missing_go_to_left,
    is_multilabel,
)


def test_thresholds_of_trees_in_double_precision():
    """Compare the decisions of the converted thresholds with scikit-learn."""
    x, y = load_iris(return_X_y=True)
    forest = RandomForestClassifier(n_estimators=20, random_state=0)
    forest.fit(x, y)
    for tree in forest.estimators_:
        thresholds = np.array(get_thresholds(tree.tree_))
        is_leaf = tree.tree_.children_left == -1
        assert np.all(thresholds[is_leaf] == -2)
        for threshold, converted in zip(
            tree.tree_.threshold[~is_leaf], thresholds[~is_leaf], strict=True
        ):
            # Doubles around the threshold and the next float32 values:
            v = np.float32(threshold)
            candidates = [threshold, float(v), converted]
            for value in list(candidates):
                for direction in (-np.inf, np.inf):
                    candidates.append(float(np.nextafter(value, direction)))
                    candidates.append(
                        float(np.nextafter(np.float32(value), direction))
                    )
            for c in candidates:
                assert (c <= converted) == (np.float32(c) <= threshold)


def test_thresholds_of_trees_with_features_on_thresholds():
    """Features which are equal to a float32 threshold, e.g. 1.9."""
    x, y = load_iris(return_X_y=True)
    for seed in range(10):
        forest = RandomForestClassifier(n_estimators=20, random_state=seed)
        forest.fit(x, y)
        for tree in forest.estimators_:
            thresholds = get_thresholds(tree.tree_)
            t = tree.tree_
            for sample in x:
                node = 0
                while t.children_left[node] != -1:
                    value = sample[t.feature[node]]
                    expected = np.float32(value) <= t.threshold[node]
                    assert (value <= thresholds[node]) == expected
                    node = (
                        t.children_left[node]
                        if expected
                        else t.children_right[node]
                    )


@pytest.mark.parametrize(
    'estimator',
    [RidgeClassifier(), MLPClassifier(max_iter=20, random_state=0)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize('target', ['binary', 'multiclass', 'multilabel'])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_is_multilabel(estimator, target: str):
    """Test the detection of classifiers fitted on a label indicator matrix."""
    if target == 'binary':
        x, y = load_breast_cancer(return_X_y=True)
    elif target == 'multiclass':
        x, y = load_iris(return_X_y=True)
    else:
        x, y = make_multilabel_classification(n_classes=3, random_state=0)
    estimator.fit(x, y)
    assert is_multilabel(estimator) == (target == 'multilabel')


@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_is_multilabel_without_label_binarizer():
    """Test the detection with an estimator without label binarizer."""
    x, y = load_diabetes(return_X_y=True)
    assert not is_multilabel(MLPRegressor(max_iter=20).fit(x, y))


def multi_output_target(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Create a target with three outputs and 3, 2 and 4 classes."""
    return np.c_[y, x[:, 0] > 5.8, np.digitize(x[:, 2], [2, 4, 5.5])]


@pytest.mark.parametrize(
    'estimator',
    [DecisionTreeClassifier(random_state=0), KNeighborsClassifier()],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize('multi_output', [False, True], ids=['single', 'multi'])
def test_output_classes(estimator, multi_output: bool):
    """Test the number of classes per output of a classifier."""
    x, y = load_iris(return_X_y=True)
    if multi_output:
        y = multi_output_target(x, y)
    estimator.fit(x, y)
    assert get_output_classes(estimator) == ([3, 2, 4] if multi_output else [3])


def test_class_fractions_of_multi_output_tree():
    """Test the concatenated class fractions of a multi-output tree."""
    x, y = load_iris(return_X_y=True)
    y = multi_output_target(x, y)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x, y).tree_
    fractions = np.array(get_class_fractions(tree))
    assert fractions.shape == (tree.node_count, 3 + 2 + 4)
    # The fractions of each output sum up to 1:
    for start, end in [(0, 3), (3, 5), (5, 9)]:
        np.testing.assert_allclose(fractions[:, start:end].sum(axis=1), 1)


@pytest.mark.parametrize(
    'params',
    [
        dict(sample_weight=np.linspace(0.1, 3.0, 150)),
        dict(class_weight={0: 1.0, 1: 3.3, 2: 0.7}),
    ],
    ids=['sample_weight', 'class_weight'],
)
def test_class_fractions_of_weighted_tree(params: dict):
    """Test the unrounded class fractions of a tree with weighted samples."""
    x, y = load_iris(return_X_y=True)
    sample_weight = params.pop('sample_weight', None)
    est = DecisionTreeClassifier(max_depth=3, random_state=0, **params)
    est.fit(x, y, sample_weight=sample_weight)
    fractions = np.array(get_class_fractions(est.tree_))
    # The fractions of the leaves are the probabilities of scikit-learn:
    leaves = est.apply(x)
    np.testing.assert_allclose(fractions[leaves], est.predict_proba(x))


def test_densify_sparse_estimator():
    """Test the conversion of the sparse matrices of a fitted estimator."""
    x, y = load_iris(return_X_y=True)
    est = KNeighborsClassifier().fit(csr_matrix(x), y)
    dense = densify(est)
    assert dense is not est
    assert not issparse(dense._fit_X)
    np.testing.assert_array_equal(dense._fit_X, x)
    # The passed estimator remains unchanged:
    assert issparse(est._fit_X)


def test_densify_dense_estimator():
    """Test an estimator without sparse matrices, which isn't copied."""
    x, y = load_iris(return_X_y=True)
    est = KNeighborsClassifier().fit(x, y)
    assert densify(est) is est


def test_has_missing_go_to_left():
    """Test the detection of splits which send missing values to the left."""
    # Node 0 is a split, nodes 1 and 2 are leaves:
    tree_data = dict(lefts=[1, -1, -1], missing_go_to_left=[0, 1, 1])
    assert not has_missing_go_to_left(tree_data)
    tree_data['missing_go_to_left'] = [1, 0, 0]
    assert has_missing_go_to_left(tree_data)
