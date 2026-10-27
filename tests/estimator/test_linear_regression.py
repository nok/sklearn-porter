import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
    SGDRegressor,
)
from sklearn.svm import LinearSVR
from sklearn.utils import get_tags

# sklearn-porter
from sklearn_porter import Estimator, can
from tests.commons import LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

ESTIMATORS = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'RidgeCV': RidgeCV(),
    'RidgeCV-alpha_per_target': RidgeCV(alpha_per_target=True),
    'Lasso': Lasso(),
    'LassoCV': LassoCV(cv=3),
    'ElasticNet': ElasticNet(),
    'ElasticNetCV': ElasticNetCV(cv=3, l1_ratio=[0.1, 0.5, 0.9]),
    'BayesianRidge': BayesianRidge(),
    'ARDRegression': ARDRegression(),
    'HuberRegressor': HuberRegressor(max_iter=1000),
    'SGDRegressor': SGDRegressor(random_state=0),
    'SGDRegressor-average': SGDRegressor(average=True, random_state=0),
    'LinearSVR': LinearSVR(random_state=0),
    'LinearSVR-squared': LinearSVR(
        loss='squared_epsilon_insensitive', random_state=0
    ),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', REGRESSION_DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('name', ESTIMATORS.keys())
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_estimator_linear_regression(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    name: str,
    fit_intercept: bool,
):
    """Test and compare different linear regressors."""
    orig_est = clone(ESTIMATORS[name]).set_params(fit_intercept=fit_intercept)

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = dataset.data, dataset.target
    if np.ndim(y) > 1 and not get_tags(orig_est).target_tags.multi_output:
        pytest.skip(
            'Skip multiple outputs, which the regressor does not support'
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_linear_regression'),
            ('estimator', name),
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
