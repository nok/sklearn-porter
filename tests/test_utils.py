import numpy as np
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_multilabel_classification,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn_porter.utils import get_thresholds, is_multilabel


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
