from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir


def drop_decision_ties(est: GradientBoostingClassifier, x: np.ndarray):
    """
    Drop the samples whose (two most likely) classes have (almost) the same
    raw prediction. Which of them is predicted depends on rounding errors.
    """
    raw = est.decision_function(x)
    if raw.ndim == 1:
        return x[~np.isclose(raw, 0, rtol=0, atol=1e-9)]
    raw = np.sort(raw, axis=1)
    return x[~np.isclose(raw[:, -1], raw[:, -2], rtol=0, atol=1e-9)]


def make(est, x: np.ndarray, tmp_dir: Path, language: str, template: str):
    try:
        return Estimator(est).make(
            x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except (exception.CodeTooLarge, exception.TooManyConstants) as e:
        pytest.skip(str(e))


# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

CLASSIFIER_PARAMS = {
    'default': dict(n_estimators=20),
    'exponential': dict(n_estimators=20, loss='exponential'),
    'init_zero': dict(n_estimators=20, init='zero'),
    'subsample': dict(n_estimators=20, subsample=0.5, max_features=0.5),
    'deep': dict(n_estimators=10, max_depth=6, learning_rate=0.5),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('params', CLASSIFIER_PARAMS)
def test_estimator_gradient_boosting_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    params: str,
):
    """Test and compare GradientBoostingClassifier classifiers."""
    orig_est = GradientBoostingClassifier(
        random_state=0, **CLASSIFIER_PARAMS[params]
    )
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = dataset.data, dataset.target
    if params == 'exponential':
        # The exponential loss supports only binary classification:
        y = (y > 0).astype(int)
    orig_est.fit(X=x, y=y)
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    test_x = drop_decision_ties(orig_est, test_x)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_gradient_boosting_classifier'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('params', params),
        ],
    )
    y_pred, y_proba = make(orig_est, test_x, tmp_dir, language, template)
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    np.testing.assert_allclose(
        y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-9
    )


REGRESSOR_PARAMS = {
    'default': dict(n_estimators=20),
    'absolute_error': dict(n_estimators=20, loss='absolute_error'),
    'huber': dict(n_estimators=20, loss='huber'),
    'quantile': dict(n_estimators=20, loss='quantile', alpha=0.8),
    'init_zero': dict(n_estimators=20, init='zero'),
    'subsample': dict(n_estimators=20, subsample=0.5, max_features=0.5),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', REGRESSOR_PARAMS)
def test_estimator_gradient_boosting_regressor(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    params: str,
):
    """Test and compare GradientBoostingRegressor regressors."""
    orig_est = GradientBoostingRegressor(
        random_state=0, **REGRESSOR_PARAMS[params]
    )
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = dataset.data, dataset.target
    if np.ndim(y) > 1:
        y = y[:, 0]  # a single output (e.g. of linnerud)
    orig_est.fit(X=x, y=y.astype(float))
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_gradient_boosting_regressor'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('params', params),
        ],
    )
    y_pred, _ = make(orig_est, test_x, tmp_dir, language, template)
    # The ported estimators add the same values in the same order:
    np.testing.assert_allclose(
        y_pred.astype(float), orig_est.predict(test_x), rtol=1e-12, atol=0
    )
