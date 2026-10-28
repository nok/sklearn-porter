from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import MaxAbsScaler

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'default': dict(),
}


def _samples(x: np.ndarray, *rows: np.ndarray) -> np.ndarray:
    """Samples with missing values, negative zeros and additional rows."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            np.full((1, x.shape[1]), -0.0),
            *rows,
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
def test_transformer_max_abs_scaler(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a MaxAbsScaler."""
    x = dataset.data
    orig_est = MaxAbsScaler(**params).fit(x)
    test_x = _samples(x, -x.max(axis=0), x.max(axis=0))
    est = Transformer(orig_est, language=language, template=template)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_max_abs_scaler'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer divides like scikit-learn, so the
    # transformed features are identical (also the missing values and the
    # signs of the zeros):
    x_true = orig_est.transform(test_x)
    np.testing.assert_array_equal(x_tr, x_true)
    np.testing.assert_array_equal(np.signbit(x_tr), np.signbit(x_true))
    assert est.test(test_x) == 1.0
