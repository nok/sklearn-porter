from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

# The composite transformers (of the first features of a dataset):
TRANSFORMERS = {
    'scalers_with_remainder': lambda: ColumnTransformer(
        [
            ('standard', StandardScaler(), [0, 1]),
            ('min_max', MinMaxScaler(), [2]),
        ],
        remainder='passthrough',
    ),
    'pipeline_with_weights': lambda: ColumnTransformer(
        [
            ('robust', make_pipeline(SimpleImputer(), RobustScaler()), [0]),
            ('drop', 'drop', [1]),
            ('k_bins', KBinsDiscretizer(n_bins=3, encode='onehot'), [2, 3]),
        ],
        transformer_weights={'robust': 2.0, 'k_bins': 0.5},
    ),
    'feature_union': lambda: FeatureUnion(
        [
            ('standard', StandardScaler()),
            ('passthrough', 'passthrough'),
            ('poly', PolynomialFeatures(2, include_bias=False)),
        ],
        transformer_weights={'standard': 3.0},
    ),
}


def _samples(x: np.ndarray) -> np.ndarray:
    """Samples with negative zeros and missing values (in the first one)."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            np.full((1, x.shape[1]), -0.0),
        )
    )
    samples[::7, 0] = np.nan
    return samples


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('name', TRANSFORMERS.keys())
def test_transformer_column_transformer(
    tmp_root_dir: Path,
    name: str,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of composite transformers."""
    x = dataset.data[:, :5]
    if x.shape[1] < 4:
        pytest.skip('The dataset has fewer than four features.')
    test_x = _samples(x)
    if name != 'pipeline_with_weights':  # without an imputer
        test_x = test_x[~np.isnan(test_x).any(axis=1)]
    orig_est = TRANSFORMERS[name]().fit(x)
    est = Transformer(orig_est, language=language, template=template)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_column_transformer'),
            ('name', name),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformers of the parts transform the selected features
    # like scikit-learn, the results are concatenated (and weighted):
    x_true = orig_est.transform(test_x)
    if hasattr(x_true, 'toarray'):
        x_true = x_true.toarray()
    assert x_tr.shape == x_true.shape
    np.testing.assert_array_equal(x_tr, x_true)
    assert est.test(test_x) == 1.0
