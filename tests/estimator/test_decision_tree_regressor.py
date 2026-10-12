from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.tree import DecisionTreeRegressor

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', REGRESSION_DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('max_depth', [3, 10, None])
@pytest.mark.parametrize(
    'criterion', ['squared_error', 'absolute_error', 'poisson']
)
@pytest.mark.parametrize('weighted', [False, True])
def test_estimator_decision_tree_regressor(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    max_depth: int | None,
    criterion: str,
    weighted: bool,
):
    """Test and compare different DecisionTree regressors."""
    orig_est = DecisionTreeRegressor(
        criterion=criterion, max_depth=max_depth, random_state=1
    )

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = dataset.data, dataset.target
    # Sample weights change the leaf values, e.g. the weighted median:
    weights = np.random.RandomState(1).rand(len(y)) if weighted else None
    orig_est.fit(X=x, y=y, sample_weight=weights)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_decision_tree_regressor'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('max_depth', str(max_depth)),
            ('criterion', criterion),
            ('weighted', str(weighted)),
        ],
    )

    y_pred, _ = est.make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    # Both reach the same leaf and return its stored value:
    np.testing.assert_array_equal(
        y_pred.astype(float), orig_est.predict(test_x)
    )
