from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import Normalizer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'max': dict(norm='max'),
    'l1': dict(norm='l1'),
    'l2': dict(norm='l2'),
}


def _samples(x: np.ndarray) -> np.ndarray:
    """Samples with zeros, negative zeros and norms close to zero."""
    return np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            -dataset_uniform_x(x, 20),
            np.full((1, x.shape[1]), -0.0),
            np.full((1, x.shape[1]), 0.0),
            np.full((1, x.shape[1]), 1e-15),
            np.full((1, x.shape[1]), 3e-15),
        )
    )


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_normalizer(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a Normalizer."""
    x = dataset.data
    orig_est = Normalizer(**params).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    test_x = _samples(x)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_normalizer'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer divides by the norm like scikit-learn (or by
    # one, if it is close to zero):
    x_true = orig_est.transform(test_x)
    if orig_est.norm == 'l2':
        # scikit-learn sums the squares with `np.einsum`, whose order
        # depends on the CPU, so the norms can differ by a few ULP:
        np.testing.assert_allclose(x_tr, x_true, rtol=1e-14, atol=0)
    else:
        # The maximum and the sum of the absolute values in the order of
        # NumPy (pairwise summation) are identical:
        np.testing.assert_array_equal(x_tr, x_true)
    np.testing.assert_array_equal(np.signbit(x_tr), np.signbit(x_true))
    assert est.test(test_x) == 1.0
