import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse

# scikit-learn
from sklearn.preprocessing import KBinsDiscretizer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'quantile': dict(n_bins=3, encode='ordinal'),
    'uniform': dict(n_bins=4, encode='ordinal', strategy='uniform'),
    'kmeans': dict(n_bins=5, encode='ordinal', strategy='kmeans'),
    'onehot': dict(n_bins=3, encode='onehot'),
    'onehot_dense': dict(n_bins=4, encode='onehot-dense', strategy='uniform'),
}


def _samples(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Samples with the edges of the bins and negative zeros."""
    edges = [e[1:-1] if len(e) > 2 else e[:0] for e in bin_edges]
    n_edges = max(len(e) for e in edges)
    rows = [
        [e[min(k, len(e) - 1)] if len(e) else 0.0 for e in edges]
        for k in range(n_edges)
    ]
    return np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            np.full((1, x.shape[1]), -0.0),
            np.array(rows).reshape(-1, x.shape[1]),
            x.min(axis=0) - 1,
            x.max(axis=0) + 1,
        )
    )


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_k_bins_discretizer(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a KBinsDiscretizer."""
    x = dataset.data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # e.g. constant features
        orig_est = KBinsDiscretizer(**params).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    test_x = _samples(x, orig_est.bin_edges_)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_k_bins_discretizer'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer counts the inner edges of the bins which are
    # lower than or equal to the feature like scikit-learn (and sets the
    # transformed feature of this bin to 1 with `encode='onehot'`):
    x_true = orig_est.transform(test_x)
    if issparse(x_true):
        x_true = x_true.toarray()
    assert x_tr.shape == x_true.shape
    np.testing.assert_array_equal(x_tr, x_true)
    assert est.test(test_x) == 1.0
