from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import OrdinalEncoder

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'default': dict(),
    'encoded_values': dict(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        encoded_missing_value=-2,
    ),
    'unknown_value_nan': dict(
        handle_unknown='use_encoded_value', unknown_value=np.nan
    ),
}


def _categories(x: np.ndarray) -> np.ndarray:
    """Categories (rounded numbers) with missing values in the first one."""
    x = np.round(x)
    x[::7, 0] = np.nan
    return x


def _samples(x: np.ndarray, unknown: bool) -> np.ndarray:
    """Samples with known (and unknown) categories and missing values."""
    samples = [
        dataset_uniform_x(x),
        np.full((1, x.shape[1]), -0.0) if (x == 0).any(axis=0).all() else x[:1],
    ]
    if unknown:
        samples.append(dataset_uniform_x(x, 20) + 0.5)
        samples.append(np.full((1, x.shape[1]), np.nan))
    samples = np.vstack(samples)
    samples[::5, 0] = np.nan
    return samples


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_ordinal_encoder(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a OrdinalEncoder."""
    x = _categories(dataset.data)
    orig_est = OrdinalEncoder(**params).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    # Unknown categories raise an error in scikit-learn by default:
    unknown = orig_est.handle_unknown == 'use_encoded_value'
    test_x = _samples(x, unknown)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_ordinal_encoder'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer returns the positions of the categories like
    # scikit-learn (and the encoded values of unknown and missing ones):
    x_true = orig_est.transform(test_x)
    np.testing.assert_array_equal(x_tr, x_true)
    assert est.test(test_x) == 1.0
