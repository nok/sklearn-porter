from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import PolynomialFeatures

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'degree_2': dict(degree=2),
    'degree_3_interaction_only': dict(
        degree=3, interaction_only=True, include_bias=False
    ),
    'degree_2_3': dict(degree=(2, 3)),
    'degree_2_3_without_bias': dict(degree=(2, 3), include_bias=False),
    'bias_only': dict(degree=0),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_polynomial_features(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a PolynomialFeatures."""
    # The first features only, because the number of transformed features
    # grows with the power of the degree:
    x = dataset.data[:, :5]
    orig_est = PolynomialFeatures(**params).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    # Samples with negative zeros and with products which overflow:
    test_x = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            np.full((1, x.shape[1]), -0.0),
            np.full((1, x.shape[1]), -1e200),
        )
    )

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_polynomial_features'),
            ('params', '-'.join(map(str, params.values()))),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer multiplies the factors in the order of
    # scikit-learn, so the products are identical:
    with np.errstate(over='ignore'):
        x_true = orig_est.transform(test_x)
    assert x_tr.shape == x_true.shape
    np.testing.assert_array_equal(x_tr, x_true)
    np.testing.assert_array_equal(np.signbit(x_tr), np.signbit(x_true))
    assert est.test(test_x) == 1.0
