import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifierClass

from sklearn_porter import exceptions as exception

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, Dataset
from tests.utils import (
    dataset_generate_x,
    dataset_uniform_x,
    drop_neighbor_ties,
    fs_mkdir,
)

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
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
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
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
