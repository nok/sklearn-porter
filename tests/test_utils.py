import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from sklearn_porter.utils import get_thresholds


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
