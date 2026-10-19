from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from sklearn_porter import exceptions as exception

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', REGRESSION_DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('n_estimators', [1, 7, 30])
@pytest.mark.parametrize('max_depth', [3, None])
@pytest.mark.parametrize(
    'criterion', ['squared_error', 'absolute_error', 'poisson']
)
@pytest.mark.parametrize(
    'clazz',
    [RandomForestRegressor, ExtraTreesRegressor],
    ids=lambda x: x.__name__,
)
def test_estimator_random_forest_regressor(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_estimators: int,
    max_depth: int | None,
    criterion: str,
    clazz: type,
):
    """Test and compare different RandomForest and ExtraTrees regressors."""
    orig_est = clazz(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=1,
    )

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_random_forest_regressor'),
            ('estimator', clazz.__name__),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('n_estimators', str(n_estimators)),
            ('max_depth', str(max_depth)),
            ('criterion', criterion),
        ],
    )

    try:
        y_pred, _ = est.make(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except (exception.CodeTooLarge, exception.TooManyConstants) as e:
        pytest.skip(str(e))
    # The trees are summed up in the same order as in scikit-learn:
    np.testing.assert_array_equal(
        y_pred.astype(float), orig_est.predict(test_x)
    )
