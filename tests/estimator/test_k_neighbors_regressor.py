from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.neighbors import KNeighborsRegressor

# sklearn-porter
from sklearn_porter import Estimator, can
from tests.commons import LANGUAGES, REGRESSION_DATASETS, Dataset
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
@pytest.mark.parametrize('dataset', REGRESSION_DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('n_neighbors', [1, 3, 10])
@pytest.mark.parametrize('weights', ['uniform', 'distance'])
@pytest.mark.parametrize(
    ('metric', 'params'),
    [
        pytest.param(m, p, id=m)
        for m, p in [
            ('euclidean', dict()),
            ('manhattan', dict(metric='manhattan')),
            ('minkowski_p_3', dict(p=3)),
            ('chebyshev', dict(metric='chebyshev')),
            ('cosine', dict(metric='cosine')),
        ]
    ],
)
def test_estimator_k_neighbors_regressor(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_neighbors: int,
    weights: str,
    metric: str,
    params: dict,
):
    """Test and compare different k-Neighbors regressors."""
    orig_est = KNeighborsRegressor(
        n_neighbors=n_neighbors, weights=weights, **params
    )

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator (the dataset `linnerud` has three outputs):
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples (some of them are training samples with the distance 0):
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    test_x = drop_neighbor_ties(orig_est, test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_k_neighbors_regressor'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('n_neighbors', str(n_neighbors)),
            ('weights', weights),
            ('metric', metric),
        ],
    )

    y_pred, _ = est.make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_allclose(
        y_pred.astype(float), orig_est.predict(test_x), rtol=1e-9, atol=1e-9
    )
