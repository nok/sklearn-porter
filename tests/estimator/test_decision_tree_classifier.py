import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter import exceptions as exception

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, LANGUAGES, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('max_depth', [5, 10, 100])
@pytest.mark.parametrize('max_leaf_nodes', [2, 3])
def test_estimator_decision_tree_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    max_depth: int,
    max_leaf_nodes: int,
):
    """Test and compare different DecisionTree classifiers."""
    orig_est = DecisionTreeClassifier(
        max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=1
    )

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_decision_tree_classifier'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('max_depth', str(max_depth)),
            ('max_leaf_nodes', str(max_leaf_nodes)),
        ],
    )

    try:
        y_pred, y_proba = est.make(
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
        np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
        # The leaves are impure because of `max_leaf_nodes`:
        np.testing.assert_allclose(
            y_proba, orig_est.predict_proba(test_x), rtol=1e-14, atol=0
        )
