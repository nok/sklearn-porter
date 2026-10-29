from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import QuantileTransformer

# sklearn-porter
from sklearn_porter import Transformer
from sklearn_porter import exceptions as exception
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'n_quantiles_10': dict(n_quantiles=10),
    'n_quantiles_100': dict(n_quantiles=100),
    'default': dict(),
}


def _samples(x: np.ndarray) -> np.ndarray:
    """Samples with the quantiles, values out of range and missing values."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            x.min(axis=0),
            x.max(axis=0),
            x.min(axis=0) - 1,
            x.max(axis=0) + 1,
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
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_quantile_transformer(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a QuantileTransformer."""
    # The first features only, because each feature has up to 1000
    # quantiles (e.g. the size of a method in Java is limited):
    x = dataset.data[:, :5]
    orig_est = QuantileTransformer(random_state=0, **params).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    test_x = _samples(x)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_quantile_transformer'),
            (
                'params',
                '-'.join(f'{k}_{v}' for k, v in params.items()) or 'default',
            ),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    try:
        x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    except (exception.CodeTooLarge, exception.TooManyConstants) as e:
        pytest.skip(str(e))
    # The ported transformer interpolates between the quantiles with the
    # operations of `np.interp`, so the transformed features are identical:
    x_true = orig_est.transform(test_x)
    np.testing.assert_array_equal(x_tr, x_true)
    is_nan = np.isnan(x_true)
    np.testing.assert_array_equal(
        np.signbit(x_tr[~is_nan]), np.signbit(x_true[~is_nan])
    )
    assert est.test(test_x) == 1.0
