from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.impute import SimpleImputer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'default': dict(),
    'median': dict(strategy='median'),
    'most_frequent': dict(strategy='most_frequent'),
    'constant': dict(strategy='constant', fill_value=-1.5),
    'keep_empty_features': dict(keep_empty_features=True),
    'missing_values': dict(missing_values=-1.0, strategy='most_frequent'),
    'add_indicator': dict(add_indicator=True),
    'add_indicator_missing_values': dict(
        add_indicator=True, missing_values=-1.0, strategy='constant'
    ),
    'empty_features': dict(empty_features=True),
    'empty_features_add_indicator': dict(
        empty_features=True, add_indicator=True, strategy='median'
    ),
}


def _with_missing_values(x: np.ndarray, missing: float) -> np.ndarray:
    """Copy of the samples with missing values in the first two features."""
    x = x.astype(np.float64)
    x[::7, 0] = missing
    x[::3, 1 % x.shape[1]] = missing
    return x


def _samples(x: np.ndarray, missing: float) -> np.ndarray:
    """Samples with missing values and negative zeros."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            np.full((1, x.shape[1]), -0.0),
        )
    )
    return _with_missing_values(samples, missing)


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_simple_imputer(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a SimpleImputer."""
    missing = params.get('missing_values', np.nan)
    x = _with_missing_values(dataset.data, missing)
    kwargs = {k: v for k, v in params.items() if k != 'empty_features'}
    if params.get('keep_empty_features') or params.get('empty_features'):
        # A feature without any observed values (scikit-learn drops it
        # without `keep_empty_features=True`):
        x[:, -1] = missing
    orig_est = SimpleImputer(**kwargs).fit(x)
    test_x = _samples(dataset.data, missing)
    est = Transformer(orig_est, language=language, template=template)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_simple_imputer'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer replaces the missing values by the
    # statistics like scikit-learn (without the dropped features and with
    # the indicators of the missing values), so the transformed features
    # are identical:
    x_true = orig_est.transform(test_x)
    assert x_tr.shape == x_true.shape
    np.testing.assert_array_equal(x_tr, x_true)
    np.testing.assert_array_equal(np.signbit(x_tr), np.signbit(x_true))
    assert est.test(test_x) == 1.0
