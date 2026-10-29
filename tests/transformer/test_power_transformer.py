from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import PowerTransformer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'yeo_johnson': dict(method='yeo-johnson'),
    'yeo_johnson_without_standardize': dict(
        method='yeo-johnson', standardize=False
    ),
    'box_cox': dict(method='box-cox'),
    'box_cox_without_standardize': dict(method='box-cox', standardize=False),
}


def _samples(x: np.ndarray, positive: bool) -> np.ndarray:
    """Samples with large and small values, zeros and missing values."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            # Box-Cox can overflow for large values, then the standardization
            # of scikit-learn raises an error:
            x.max(axis=0) * (2 if positive else 100),
            x.min(axis=0) / 2 if positive else x.min(axis=0) * 100,
        )
    )
    if not positive:
        samples = np.vstack(
            (samples, np.full((1, x.shape[1]), -0.0), -dataset_uniform_x(x, 20))
        )
    samples[::7, 0] = np.nan
    return samples


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_power_transformer(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a PowerTransformer."""
    positive = params['method'] == 'box-cox'
    x = dataset.data
    if positive:  # Box-Cox requires strictly positive features
        x = x - x.min(axis=0) + 1.0
    try:
        orig_est = PowerTransformer(**params).fit(x)
    except ValueError as e:  # e.g. Box-Cox of constant features
        pytest.skip(str(e))
    est = Transformer(orig_est, language=language, template=template)
    test_x = _samples(x, positive)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_power_transformer'),
            ('params', '-'.join(f'{k}_{v}' for k, v in params.items())),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    with np.errstate(all='ignore'):
        x_true = orig_est.transform(test_x)
    # The ported transformer computes the formulas of SciPy, but `log`,
    # `exp`, `log1p` and `expm1` of some languages (and of NumPy, depending
    # on the CPU) can differ by a few ULP:
    np.testing.assert_allclose(x_tr, x_true, rtol=1e-12, atol=1e-12)
    with np.errstate(all='ignore'):
        assert est.test(test_x) == 1.0
