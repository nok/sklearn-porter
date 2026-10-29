from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import TargetEncoder

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'default': dict(),
    'smooth': dict(smooth=5.0, cv=3),
    'continuous': dict(target_type='continuous'),
}


def _categories(x: np.ndarray) -> np.ndarray:
    """Categories (rounded numbers) with missing values in the first one."""
    x = np.round(x[:, :4])
    x[::7, 0] = np.nan
    return x


def _samples(x: np.ndarray) -> np.ndarray:
    """Samples with known and unknown categories and missing values."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            np.full((1, x.shape[1]), -0.0),
            dataset_uniform_x(x, 20) + 0.5,
            np.full((1, x.shape[1]), np.nan),
        )
    )
    samples[::5, 0] = np.nan
    return samples


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_target_encoder(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a TargetEncoder."""
    # The first features only, because each category has an encoding (per
    # class of the target):
    x = _categories(dataset.data)
    y = dataset.target
    if y.ndim > 1:  # multiple outputs aren't supported by scikit-learn
        y = y[:, 0]
    orig_est = TargetEncoder(random_state=0, **params).fit(x, y)
    est = Transformer(orig_est, language=language, template=template)
    test_x = _samples(x)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_target_encoder'),
            (
                'params',
                '-'.join(f'{k}_{v}' for k, v in params.items()) or 'default',
            ),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer returns the encodings of the categories (and
    # the mean of the target for unknown categories) like scikit-learn:
    x_true = orig_est.transform(test_x)
    assert x_tr.shape == x_true.shape
    np.testing.assert_array_equal(x_tr, x_true)
    assert est.test(test_x) == 1.0
