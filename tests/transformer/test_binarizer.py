from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import Binarizer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'default': dict(),
    'threshold': dict(threshold=1.5),
}


def _samples(x: np.ndarray, *rows: np.ndarray) -> np.ndarray:
    """Samples with negative zeros and additional rows.

    The `Binarizer` of scikit-learn doesn't accept missing values.
    """
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            np.full((1, x.shape[1]), -0.0),
            *rows,
        )
    )
    return samples


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_binarizer(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a Binarizer."""
    x = dataset.data
    orig_est = Binarizer(**params).fit(x)
    # Samples on the threshold:
    test_x = _samples(x, np.full(x.shape[1], orig_est.threshold))
    est = Transformer(orig_est, language=language, template=template)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_binarizer'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer compares like scikit-learn, so the
    # transformed features are identical:
    x_true = orig_est.transform(test_x)
    np.testing.assert_array_equal(x_tr, x_true)
    np.testing.assert_array_equal(np.signbit(x_tr), np.signbit(x_true))
    assert est.test(test_x) == 1.0
