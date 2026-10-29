import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (
    RFE,
    RFECV,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import get_tags

# sklearn-porter
from sklearn_porter import Transformer
from sklearn_porter import exceptions as exception
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


def _tree(is_clf: bool):
    if is_clf:
        return DecisionTreeClassifier(max_depth=4, random_state=0)
    return DecisionTreeRegressor(max_depth=4, random_state=0)


def _score(is_clf: bool) -> Callable:
    return f_classif if is_clf else f_regression


# The feature selectors (of a classification or a regression):
SELECTORS: dict[str, Callable] = {
    # The default threshold drops the constant features (e.g. of digits):
    'variance_threshold': lambda is_clf: VarianceThreshold(),
    'select_k_best': lambda is_clf: SelectKBest(_score(is_clf), k=2),
    'select_percentile': lambda is_clf: SelectPercentile(
        _score(is_clf), percentile=50
    ),
    'generic_univariate_select': lambda is_clf: GenericUnivariateSelect(
        _score(is_clf), mode='fpr', param=0.01
    ),
    'select_fpr': lambda is_clf: SelectFpr(_score(is_clf), alpha=0.01),
    'select_fdr': lambda is_clf: SelectFdr(_score(is_clf), alpha=0.01),
    'select_fwe': lambda is_clf: SelectFwe(_score(is_clf), alpha=0.01),
    'select_from_model': lambda is_clf: SelectFromModel(
        LogisticRegression(max_iter=200) if is_clf else Ridge()
    ),
    'rfe': lambda is_clf: RFE(_tree(is_clf), n_features_to_select=2),
    'rfecv': lambda is_clf: RFECV(_tree(is_clf), step=0.2, cv=3),
    'sequential_feature_selector': lambda is_clf: SequentialFeatureSelector(
        _tree(is_clf), n_features_to_select=2, cv=3
    ),
}


def _samples(x: np.ndarray, with_missing_values: bool) -> np.ndarray:
    """Samples with negative zeros (and missing and infinite values)."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            np.full((1, x.shape[1]), -0.0),
        )
    )
    if with_missing_values:  # if the selector accepts them
        samples[::7, 0] = np.nan
        samples[1::7, -1] = np.inf
        samples[2::7, -1] = -np.inf
    return samples


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('name', SELECTORS.keys())
def test_transformer_feature_selector(
    tmp_root_dir: Path,
    name: str,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the selected features of a feature selector."""
    is_clf = dataset in DATASETS
    x, y = dataset.data, dataset.target
    if y.ndim > 1:  # the first output of a multi-output regression
        y = y[:, 0]
    with warnings.catch_warnings():
        # E.g. constant features of the univariate tests:
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)
        orig_est = SELECTORS[name](is_clf).fit(x, y)
    if not orig_est.get_support().any():
        pytest.skip('No feature is selected.')
    est = Transformer(orig_est, language=language, template=template)
    # E.g. a `VarianceThreshold` or an `RFE` of a tree accepts missing and
    # infinite values like scikit-learn:
    test_x = _samples(x, get_tags(orig_est).input_tags.allow_nan)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_feature_selector'),
            ('name', name),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported selector returns the selected features unchanged (also
    # negative zeros, missing and infinite values):
    x_true = orig_est.transform(test_x)
    np.testing.assert_array_equal(x_tr, x_true)
    is_nan = np.isnan(x_true)
    np.testing.assert_array_equal(
        np.signbit(x_tr[~is_nan]), np.signbit(x_true[~is_nan])
    )
    assert est.test(test_x) == 1.0


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_invalid_values_of_feature_selector(
    tmp_root_dir: Path, value: float, language: str, template: str
):
    """Like scikit-learn the selector rejects missing and infinite values."""
    x, y = load_iris(return_X_y=True)
    orig_est = SelectKBest(k=2).fit(x, y)
    assert not get_tags(orig_est).input_tags.allow_nan
    est = Transformer(orig_est, language=language, template=template)
    # Also a value of a feature which isn't selected:
    test_x = x[:1].copy()
    test_x[0, 0] = value
    with pytest.raises(ValueError, match='Input X contains'):
        orig_est.transform(test_x)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'invalid_values_of_feature_selector'),
            ('value', str(value)),
            ('language', language),
            ('template', template),
        ],
    )
    with pytest.raises(exception.ExecutionFailed, match='NaN or infinity'):
        est.make(test_x, directory=tmp_dir, delete_created_files=False)
    np.testing.assert_array_equal(est.make(x[:5]), orig_est.transform(x[:5]))
