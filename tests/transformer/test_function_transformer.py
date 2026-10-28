from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import FunctionTransformer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'identity': dict(func=None),
    'abs': dict(func=np.abs),
    'sqrt': dict(func=np.sqrt),
    'square': dict(func=np.square),
    'exp': dict(func=np.exp),
    'expm1': dict(func=np.expm1),
    'log': dict(func=np.log),
    'log1p': dict(func=np.log1p),
}


def _samples(x: np.ndarray) -> np.ndarray:
    """Samples with negative, small, large and missing values and zeros."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            -dataset_uniform_x(x, 20),
            dataset_uniform_x(x, 20) * 1e-10,
            np.full((1, x.shape[1]), -0.0),
            np.full((1, x.shape[1]), 0.0),
            np.full((1, x.shape[1]), -1.0),
            np.full((1, x.shape[1]), 1000.0),
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
def test_transformer_function_transformer(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a FunctionTransformer."""
    x = dataset.data
    orig_est = FunctionTransformer(**params).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    test_x = _samples(x)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_function_transformer'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    with np.errstate(all='ignore'):
        x_true = orig_est.transform(test_x)
    name = orig_est.func.__name__ if orig_est.func else 'identity'
    if name in ('absolute', 'identity', 'sqrt', 'square'):
        # The results are correctly rounded (also the signs of the zeros):
        np.testing.assert_array_equal(x_tr, x_true)
        np.testing.assert_array_equal(np.signbit(x_tr), np.signbit(x_true))
    else:
        # NumPy computes `exp` and `log` with SIMD instructions depending on
        # the CPU, which can differ by a few ULP from the target languages.
        # Go returns zero instead of subnormal numbers, e.g. of `exp(-744)`:
        tiny = np.finfo(np.float64).tiny
        np.testing.assert_allclose(x_tr, x_true, rtol=1e-14, atol=tiny)
    with np.errstate(all='ignore'):
        assert est.test(test_x) == 1.0
