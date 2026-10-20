from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.Estimator import Estimator, can
from tests.commons import (
    DATASETS,
    LANGUAGES,
    PORTER_N_GEN_REGRESSION_TESTS,
    REGRESSION_DATASETS,
    Dataset,
)
from tests.utils import dataset_uniform_x, fs_mkdir


def drop_decision_ties(est: HistGradientBoostingClassifier, x: np.ndarray):
    """
    Drop the samples whose (two most likely) classes have (almost) the same
    raw prediction. Which of them is predicted depends on rounding errors.
    """
    raw = est.decision_function(x)
    if raw.ndim == 1:
        return x[~np.isclose(raw, 0, rtol=0, atol=1e-9)]
    raw = np.sort(raw, axis=1)
    return x[~np.isclose(raw[:, -1], raw[:, -2], rtol=0, atol=1e-9)]


def prepare(x: np.ndarray, kind: str) -> tuple[np.ndarray, list | None]:
    """
    Prepare the features of a dataset.

    `missing` replaces 10 % of the features by missing values (NaN),
    `categorical` appends two categorical features with negative and
    non-integer categories and missing values.
    """
    rng = np.random.default_rng(0)
    x = np.array(x, dtype=float)
    if kind == 'missing':
        x[rng.random(x.shape) < 0.1] = np.nan
        return x, None
    if kind == 'categorical':
        bins = np.quantile(x[:, 0], [0.2, 0.4, 0.6, 0.8])
        first = np.digitize(x[:, 0], bins) - 2.0
        second = rng.integers(0, 6, len(x)) * 0.5 + 10
        cats = np.column_stack((first, second))
        cats[rng.random(cats.shape) < 0.05] = np.nan
        n = x.shape[1]
        return np.hstack((x, cats)), [n, n + 1]
    return x, None


def samples(x: np.ndarray, kind: str, categorical: list | None):
    """
    Pick training samples and generate random samples. The generated samples
    of the kind `missing` have missing values (NaN), those of the kind
    `categorical` have known and unknown categories.
    """
    rng = np.random.default_rng(1)
    generated = rng.uniform(
        np.nanmin(x, axis=0),
        np.nanmax(x, axis=0),
        size=(PORTER_N_GEN_REGRESSION_TESTS, x.shape[1]),
    )
    if kind == 'missing':
        generated[rng.random(generated.shape) < 0.1] = np.nan
    if categorical:
        generated[:, categorical] = np.round(generated[:, categorical] * 4) / 4
    return np.vstack((dataset_uniform_x(x), generated))


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
    'default': ('numeric', dict(max_iter=20)),
    'missing': ('missing', dict(max_iter=20)),
    'categorical': ('categorical', dict(max_iter=20)),
    'small': (
        'numeric',
        dict(max_iter=10, max_leaf_nodes=8, learning_rate=0.5),
    ),
    'regularized': (
        'missing',
        dict(max_iter=20, l2_regularization=1.0, max_features=0.5),
    ),
    'early_stopping': (
        'numeric',
        dict(max_iter=200, early_stopping=True, n_iter_no_change=3),
    ),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('params', CLASSIFIER_PARAMS)
def test_estimator_hist_gradient_boosting_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    params: str,
):
    """Test and compare HistGradientBoostingClassifier classifiers."""
    kind, kwargs = CLASSIFIER_PARAMS[params]
    orig_est = HistGradientBoostingClassifier(random_state=0, **kwargs)
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, categorical = prepare(dataset.data, kind)
    orig_est.set_params(categorical_features=categorical)
    orig_est.fit(X=x, y=dataset.target)
    test_x = drop_decision_ties(orig_est, samples(x, kind, categorical))

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_hist_gradient_boosting_classifier'),
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
    'default': ('numeric', dict(max_iter=20)),
    'missing': ('missing', dict(max_iter=20)),
    'categorical': ('categorical', dict(max_iter=20)),
    'absolute_error': ('numeric', dict(max_iter=20, loss='absolute_error')),
    'quantile': ('missing', dict(max_iter=20, loss='quantile', quantile=0.8)),
    'poisson': ('categorical', dict(max_iter=20, loss='poisson')),
    'gamma': ('missing', dict(max_iter=20, loss='gamma')),
    'early_stopping': (
        'numeric',
        dict(max_iter=200, early_stopping=True, n_iter_no_change=3),
    ),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', REGRESSOR_PARAMS)
def test_estimator_hist_gradient_boosting_regressor(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    params: str,
):
    """Test and compare HistGradientBoostingRegressor regressors."""
    kind, kwargs = REGRESSOR_PARAMS[params]
    orig_est = HistGradientBoostingRegressor(random_state=0, **kwargs)
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, categorical = prepare(dataset.data, kind)
    y = np.asarray(dataset.target, dtype=float)
    if y.ndim > 1:
        y = y[:, 0]  # a single output (e.g. of linnerud)
    if kwargs.get('loss') == 'gamma':
        y = y + 1.0  # the gamma loss requires positive targets
    orig_est.set_params(categorical_features=categorical)
    orig_est.fit(X=x, y=y)
    test_x = samples(x, kind, categorical)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_hist_gradient_boosting_regressor'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('params', params),
        ],
    )
    y_pred, _ = make(orig_est, test_x, tmp_dir, language, template)
    # The ported estimators add the same values in the same order, only the
    # exponential function of the log link can differ in the last digit:
    np.testing.assert_allclose(
        y_pred.astype(float), orig_est.predict(test_x), rtol=1e-12, atol=0
    )
