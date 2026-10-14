from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.base import is_classifier
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, Dataset
from tests.utils import fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

CLASSIFIERS = [
    DecisionTreeClassifier(max_depth=6, random_state=0),
    RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0),
    ExtraTreesClassifier(n_estimators=5, max_depth=5, random_state=0),
]

REGRESSORS = [
    DecisionTreeRegressor(max_depth=6, random_state=0),
    RandomForestRegressor(n_estimators=5, max_depth=5, random_state=0),
    ExtraTreesRegressor(n_estimators=5, max_depth=5, random_state=0),
]


def with_missing_values(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Replace 20 % of the feature values by missing values (NaN)."""
    x = x.copy()
    x[np.random.RandomState(seed).rand(*x.shape) < 0.2] = np.nan
    return x


def make(
    orig_est,
    x: np.ndarray,
    tmp_root_dir: Path,
    name: str,
    language: str,
    template: str,
):
    """Make the predictions of the ported estimator."""
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_tree_missing_values'),
            ('estimator', name),
            ('language', language),
            ('template', template),
        ],
    )
    try:
        return Estimator(orig_est).make(
            x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except exception.CodeTooLarge:
        pytest.skip('The code is too large to compile (e.g. Java attached)')


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize(
    'orig_est', CLASSIFIERS, ids=lambda x: type(x).__name__
)
def test_tree_classifier_with_missing_values(
    tmp_root_dir: Path,
    orig_est,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test the routing of missing values (NaN) of tree classifiers."""
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # scikit-learn routes missing values via `missing_go_to_left`, even if
    # the estimator was fitted without missing values:
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    test_x = with_missing_values(x)

    name = f'{type(orig_est).__name__}_{dataset.name}'
    y_pred, y_proba = make(
        orig_est, test_x, tmp_root_dir, name, language, template
    )
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    np.testing.assert_allclose(
        y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-5
    )


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('orig_est', REGRESSORS, ids=lambda x: type(x).__name__)
def test_tree_regressor_with_missing_values(
    tmp_root_dir: Path,
    orig_est,
    language: str,
    template: str,
):
    """Test the routing of missing values (NaN) of tree regressors."""
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    x, y = load_diabetes(return_X_y=True)
    orig_est.fit(X=x, y=y)
    test_x = with_missing_values(x[:150])

    name = type(orig_est).__name__
    y_pred, _ = make(orig_est, test_x, tmp_root_dir, name, language, template)
    np.testing.assert_allclose(y_pred, orig_est.predict(test_x), rtol=1e-9)


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize(
    'orig_est', CLASSIFIERS + REGRESSORS, ids=lambda x: type(x).__name__
)
def test_tree_fitted_with_missing_values(
    tmp_root_dir: Path,
    orig_est,
    language: str,
    template: str,
):
    """Test trees which were fitted on data with missing values."""
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    x, y = load_diabetes(return_X_y=True)
    if is_classifier(orig_est):
        y = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    orig_est.fit(X=with_missing_values(x), y=y)

    # The best splitter (not the random splitter of extra trees) separates
    # the samples with missing values with an infinite threshold:
    trees = getattr(orig_est, 'estimators_', [orig_est])
    thresholds = np.concatenate([t.tree_.threshold for t in trees])
    is_extra = isinstance(orig_est, (ExtraTreesClassifier, ExtraTreesRegressor))
    assert np.isinf(thresholds).any() != is_extra

    test_x = with_missing_values(x[:150], seed=1)
    name = f'{type(orig_est).__name__}_fitted'
    y_pred, _ = make(orig_est, test_x, tmp_root_dir, name, language, template)
    if is_classifier(orig_est):
        np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    else:
        np.testing.assert_allclose(y_pred, orig_est.predict(test_x), rtol=1e-9)
