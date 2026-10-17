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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn_porter import exceptions as exception
from sklearn_porter.language.c import C
from sklearn_porter.language.go import Go
from sklearn_porter.language.java import Java
from sklearn_porter.language.js import JavaScript
from sklearn_porter.language.php import PHP
from sklearn_porter.language.ruby import Ruby
from sklearn_porter.utils import (
    densify,
    dump_model_data,
    get_class_fractions,
    get_labels,
    get_leaf_values,
    get_output_classes,
    get_thresholds,
    get_tree_data,
    has_infinite_thresholds,
    has_missing_go_to_left,
    is_multilabel,
    with_infinity,
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
    # The model data contains only the splits:
    assert not has_missing_go_to_left(dict(missing_go_to_left=[0, 0]))
    assert has_missing_go_to_left(dict(missing_go_to_left=[0, 1]))


def _find_leaf(tree_data: dict, x: np.ndarray) -> int:
    """Find the leaf of a sample like the ported trees."""
    node = 0
    while node >= 0:
        is_left = x[tree_data['indices'][node]] <= tree_data['thresholds'][node]
        node = tree_data['lefts' if is_left else 'rights'][node]
    return -node - 1


@pytest.mark.parametrize(
    'orig_est',
    [
        DecisionTreeClassifier(max_depth=3, random_state=0),
        DecisionTreeClassifier(random_state=0),
        DecisionTreeRegressor(random_state=0),
    ],
    ids=['shallow', 'deep', 'regressor'],
)
def test_tree_data(orig_est):
    """Test the compact model data of a tree, which contains only the
    splits and the values of the leaves."""
    x, y = load_iris(return_X_y=True)
    tree = orig_est.fit(x, y).tree_
    values = tree.value.reshape(tree.node_count, -1)
    tree_data = get_tree_data(tree, values, 'values')
    is_split = tree.children_left != -1
    assert len(tree_data['lefts']) == is_split.sum()
    assert len(tree_data['values']) == tree.n_leaves * values.shape[1]
    # Each leaf is referenced by exactly one split:
    refs = sorted(tree_data['lefts'] + tree_data['rights'])
    assert refs[: tree.n_leaves] == list(range(-tree.n_leaves, 0))
    # The ported trees find the same leaves as scikit-learn:
    leaf_values = get_leaf_values(tree_data, 'values')
    assert len(leaf_values) == tree.n_leaves
    expected = values[tree.apply(x.astype(np.float32))]
    found = [leaf_values[_find_leaf(tree_data, sample)] for sample in x]
    np.testing.assert_array_equal(found, expected)


def test_tree_data_of_single_leaf():
    """Test the compact model data of a tree which consists of a leaf."""
    x, _ = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier().fit(x, np.zeros(len(x))).tree_
    assert tree.node_count == 1
    tree_data = get_tree_data(tree, get_class_fractions(tree), 'classes')
    # A split whose children are both the leaf, so no array is empty:
    assert tree_data == dict(
        lefts=[-1],
        rights=[-1],
        thresholds=[0.0],
        indices=[0],
        missing_go_to_left=[0],
        classes=[1.0],
    )
    assert get_leaf_values(tree_data, 'classes') == [[1.0]]
    assert _find_leaf(tree_data, x[0]) == 0


def test_infinite_values():
    """Test the conversion of infinite values to literals and JSON."""
    convert = with_infinity(str, 'INFINITY')
    assert convert(np.inf) == 'INFINITY'
    assert convert(-np.inf) == '-INFINITY'
    assert convert(np.float64(1.5)) == '1.5'
    assert convert(2) == '2'
    # JSON doesn't support infinite values, but most parsers read numbers
    # which are too large for a double as infinite values:
    data = dict(thresholds=[np.inf, -np.inf, 1.5])
    assert dump_model_data(data) == '{"thresholds":[1e999,-1e999,1.5]}'
    assert has_infinite_thresholds(data)
    assert not has_infinite_thresholds(dict(thresholds=[-2.0, 1.5]))


@pytest.mark.parametrize(
    ('y', 'expected'),
    [
        ([2, 0, 5], ([0, 2, 5], 'int')),
        ([-1.0, 1.0, 1.0], ([-1.0, 1.0], 'double')),
        (['b', 'a', 'a'], (['a', 'b'], 'str')),
        ([True, False, True], ([0, 1], 'int')),
        (np.array(['b', 'a'], dtype=object), (['a', 'b'], 'str')),
    ],
    ids=['int', 'float', 'str', 'bool', 'object'],
)
def test_labels(y: list, expected: tuple):
    """Test the labels of the classes and their kind."""
    x = np.arange(len(y)).reshape(-1, 1)
    labels, kind = get_labels(DecisionTreeClassifier().fit(x, y))
    assert (labels, kind) == expected
    assert all(type(label) is type(expected[0][0]) for label in labels)


def test_labels_of_multiple_outputs():
    """Test the concatenated labels of all outputs."""
    x = np.arange(4).reshape(-1, 1)
    y = np.array([['a', 'x'], ['b', 'y'], ['c', 'x'], ['a', 'y']])
    est = DecisionTreeClassifier().fit(x, y)
    assert get_labels(est) == (['a', 'b', 'c', 'x', 'y'], 'str')


@pytest.mark.parametrize(
    'y',
    [
        np.array([1, 'a'], dtype=object),
        np.array([1, 2.5], dtype=object),
        [0, 2**31],
    ],
    ids=['mixed-int-str', 'mixed-int-float', 'int64'],
)
def test_unsupported_labels(y):
    """Test the labels of different or unsupported types."""
    est = DecisionTreeClassifier()
    est.classes_ = np.asarray(y)
    with pytest.raises(exception.NotSupportedYetError):
        get_labels(est)


@pytest.mark.parametrize(
    ('language', 'expected'),
    [
        (C, '"a\\042b\\134c\\077\'\\303\\244"'),
        (Go, '"a\\U00000022b\\U0000005cc?\'\\U000000e4"'),
        (Java, '"a\\042b\\134c?\'\\u00e4"'),
        (JavaScript, '"a\\"b\\\\c?\'\\u00e4"'),
        (PHP, "'a\"b\\\\c?\\'ä'"),
        (Ruby, "'a\"b\\\\c?\\'ä'"),
    ],
    ids=lambda x: getattr(x, 'KEY', None),
)
def test_string_literals(language, expected: str):
    """Test the escapes of the string literals of all languages."""
    assert language.str_literal('a"b\\c?\'ä') == expected
