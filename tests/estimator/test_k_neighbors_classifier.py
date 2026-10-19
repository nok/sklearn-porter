import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifierClass

from sklearn_porter import exceptions as exception

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, LANGUAGES, Dataset
from tests.utils import (
    dataset_generate_x,
    dataset_uniform_x,
    drop_neighbor_ties,
    fs_mkdir,
)

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('n_neighbors', [1, 2, 3, 5])
@pytest.mark.parametrize(
    ('metric', 'params'),
    [
        pytest.param(m, p, id=m)
        for m, p in [
            ('euclidean', dict()),
            ('manhattan', dict(metric='manhattan')),
            ('minkowski_p_3', dict(p=3)),
            ('minkowski_p_0_5', dict(p=0.5)),
            ('chebyshev', dict(metric='chebyshev')),
            ('cosine', dict(metric='cosine')),
        ]
    ],
)
def test_estimator_k_neighbors_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_neighbors: int,
    metric: str,
    params: dict,
):
    """Test and compare different k-Neighbors classifiers."""
    orig_est = KNeighborsClassifierClass(n_neighbors=n_neighbors, **params)

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    test_x = drop_neighbor_ties(orig_est, test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_k_neighbors_classifier'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('n_neighbors', str(n_neighbors)),
            ('metric', metric),
        ],
    )

    try:
        score = est.test(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except exception.CodeTooLarge:
        msg = (
            'Code too large for the combination: '
            f'language: {language}, template: {template}, dataset: {dataset.name}'
        )
        warnings.warn(msg, stacklevel=2)
    except Exception as e:
        pytest.fail('Unexpected exception ... ' + str(e))
    else:
        assert score == 1.0


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('n_neighbors', [1, 3])
def test_estimator_k_neighbors_classifier_with_hamming_metric(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_neighbors: int,
):
    """Test and compare k-Neighbors classifiers with the Hamming distance."""
    orig_est = KNeighborsClassifierClass(
        n_neighbors=n_neighbors, metric='hamming'
    )

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    # The Hamming distance is meant for categorical data, so binarize the
    # features, otherwise (almost) all distances would be 1:
    x = (dataset.data > np.median(dataset.data, axis=0)).astype(float)
    y = dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    # Most distances are equal, so many samples are dropped as neighbor ties:
    rng = np.random.RandomState(0)
    test_x = np.vstack(
        (dataset_uniform_x(x), rng.randint(2, size=(45, x.shape[1])))
    ).astype(float)
    test_x = drop_neighbor_ties(orig_est, test_x)
    assert len(test_x) > 0

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_k_neighbors_classifier_with_hamming_metric'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('n_neighbors', str(n_neighbors)),
        ],
    )

    try:
        score = est.test(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except exception.CodeTooLarge:
        msg = (
            'Code too large for the combination: '
            f'language: {language}, template: {template}, dataset: {dataset.name}'
        )
        warnings.warn(msg, stacklevel=2)
    else:
        assert score == 1.0


@pytest.mark.parametrize('template', ['attached', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('weights', ['uniform', 'distance'])
@pytest.mark.parametrize('n_outputs', [1, 2])
@pytest.mark.parametrize('n_neighbors', [1, 2, 5, 30, 120])
def test_estimator_k_neighbors_classifier_with_tied_neighbors(
    tmp_root_dir: Path,
    n_neighbors: int,
    n_outputs: int,
    weights: str,
    language: str,
    template: str,
):
    """Test the tied neighbors of k-Neighbors classifiers: of several
    training samples with the same distance, the ported estimators pick the
    first ones in the order of the training data (in all languages), also if
    all training samples are neighbors (`n_neighbors=120`)."""
    # Discrete features, so many training samples have the same distance:
    rng = np.random.RandomState(0)
    x = rng.randint(3, size=(120, 2)).astype(float)
    y = rng.randint(3, size=(120, n_outputs))
    test_x = rng.randint(3, size=(40, 2)).astype(float)
    orig_est = KNeighborsClassifierClass(
        n_neighbors=n_neighbors, weights=weights
    )
    orig_est.fit(X=x, y=y if n_outputs > 1 else y.ravel())

    # Expected class probabilities: the (weighted) votes of the first
    # `n_neighbors` training samples, sorted (stable) by their distances:
    dists = np.sqrt(((test_x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    neighbors = np.argsort(dists, axis=1, kind='stable')[:, :n_neighbors]
    votes = np.ones(neighbors.shape)
    if weights == 'distance':
        # Only the neighbors with the distance 0 vote, if there are any:
        with np.errstate(divide='ignore'):
            votes = 1.0 / np.take_along_axis(dists, neighbors, axis=1)
        is_exact = np.isinf(votes)
        votes = np.where(is_exact.any(axis=1, keepdims=True), is_exact, votes)
    exp_probas = [
        np.array(
            [
                np.bincount(y[n, j], weights=v, minlength=3) / v.sum()
                for n, v in zip(neighbors, votes, strict=True)
            ]
        )
        for j in range(n_outputs)
    ]
    exp_classes = np.column_stack([p.argmax(axis=1) for p in exp_probas])

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_k_neighbors_classifier_with_tied_neighbors'),
            ('language', language),
            ('template', template),
            ('n_neighbors', str(n_neighbors)),
            ('n_outputs', str(n_outputs)),
            ('weights', weights),
        ],
    )

    y_classes, y_probas = Estimator(orig_est).make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    if n_outputs == 1:
        y_classes, y_probas = np.asarray(y_classes)[:, None], [y_probas]
    np.testing.assert_array_equal(y_classes, exp_classes)
    for y_proba, exp_proba in zip(y_probas, exp_probas, strict=True):
        np.testing.assert_allclose(y_proba, exp_proba)


@pytest.mark.parametrize('template', ['attached', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('n_neighbors', [1, 5])
@pytest.mark.parametrize(
    ('metric', 'params'),
    [
        pytest.param(m, p, id=m)
        for m, p in [
            ('euclidean', dict()),
            ('manhattan', dict(metric='manhattan')),
            ('chebyshev', dict(metric='chebyshev')),
            ('cosine', dict(metric='cosine')),
        ]
    ],
)
def test_estimator_k_neighbors_classifier_with_distance_weights(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_neighbors: int,
    metric: str,
    params: dict,
):
    """Test and compare k-Neighbors classifiers whose neighbors vote with
    the inverse of their distances (`weights='distance'`)."""
    orig_est = KNeighborsClassifierClass(
        n_neighbors=n_neighbors, weights='distance', **params
    )

    # Estimator:
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples, incl. training samples with the distance 0 to a neighbor:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x), x[::15]))
    test_x = drop_neighbor_ties(orig_est, test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_k_neighbors_classifier_with_distance_weights'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('n_neighbors', str(n_neighbors)),
            ('metric', metric),
        ],
    )

    try:
        y_pred, y_proba = est.make(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except exception.CodeTooLarge:
        pytest.skip('The code is too large to compile (e.g. Java attached)')
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    np.testing.assert_allclose(
        y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-5
    )


@pytest.mark.parametrize('template', ['attached', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('n_outputs', [1, 2])
def test_estimator_k_neighbors_classifier_with_zero_distances(
    tmp_root_dir: Path,
    n_outputs: int,
    language: str,
    template: str,
):
    """Test the votes of neighbors with the distance 0 and distance weights:
    only these neighbors vote with the weight 1 (like scikit-learn)."""
    # Duplicated training samples with different classes:
    x = np.array(
        [[0, 0], [0, 0], [0, 0], [1, 0], [0, 3], [4, 4], [5, 4], [4, 6]],
        dtype=float,
    )
    y = np.array([[0, 1, 1, 2, 2, 0, 1, 2], [1, 0, 1, 0, 1, 0, 1, 0]]).T
    y = y[:, :n_outputs] if n_outputs > 1 else y[:, 0]
    orig_est = KNeighborsClassifierClass(n_neighbors=4, weights='distance')
    orig_est.fit(X=x, y=y)

    # Samples with three, one or no neighbors with the distance 0:
    test_x = np.array([[0, 0], [1, 0], [4, 4], [0.5, 0], [4.5, 4.5]])
    assert len(drop_neighbor_ties(orig_est, test_x)) == len(test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_k_neighbors_classifier_with_zero_distances'),
            ('language', language),
            ('template', template),
            ('n_outputs', str(n_outputs)),
        ],
    )

    y_pred, y_proba = Estimator(orig_est).make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    expected = orig_est.predict_proba(test_x)
    if n_outputs == 1:
        y_proba, expected = [y_proba], [expected]
    for actual, desired in zip(y_proba, expected, strict=True):
        np.testing.assert_allclose(actual, desired, rtol=0, atol=1e-12)
