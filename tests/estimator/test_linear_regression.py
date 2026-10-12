from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', REGRESSION_DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize(
    'estimator',
    [LinearRegression, Ridge, Lasso, ElasticNet],
    ids=lambda x: x.__name__,
)
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_estimator_linear_regression(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    estimator: type,
    fit_intercept: bool,
):
    """Test and compare different linear regressors."""
    orig_est = estimator(fit_intercept=fit_intercept)

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
            ('test', 'estimator_linear_regression'),
            ('estimator', estimator.__name__),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('fit_intercept', str(fit_intercept)),
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
        y_pred.astype(float), orig_est.predict(test_x), rtol=1e-6, atol=1e-8
    )
