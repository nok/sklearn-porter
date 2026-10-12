import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifierClass

from sklearn_porter import exceptions as exception

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, Dataset
from tests.utils import (
    dataset_generate_x,
    dataset_uniform_x,
    drop_neighbor_ties,
    fs_mkdir,
)

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('n_neighbors', [1, 2, 3, 5])
def test_estimator_k_neighbors_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_neighbors: int,
):
    """Test and compare different k-Neighbors classifiers."""
    orig_est = KNeighborsClassifierClass(n_neighbors=n_neighbors)

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    test_x = drop_neighbor_ties(orig_est, test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_decision_tree_classifier'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('n_neighbors', str(n_neighbors)),
        ],
    )

    try:
        score = est.test(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except exception.CodeTooLarge:
        msg = (
            'Code too large for the combination: '
            f'language: {language}, template: {template}, dataset: {dataset.name}'
        )
        warnings.warn(msg, stacklevel=2)
    except Exception as e:
        pytest.fail('Unexpected exception ... ' + str(e))
    else:
        assert score == 1.0
