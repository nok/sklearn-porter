from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.Estimator import Estimator, can
from tests.commons import LANGUAGES
from tests.utils import (
    dataset_generate_x,
    dataset_uniform_x,
    drop_neighbor_ties,
    fs_mkdir,
)

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


def load_multi_output_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset with a target of several outputs."""
    if name == 'iris':
        # Three outputs with 3, 2 and 4 classes:
        x, y = load_iris(return_X_y=True)
        return x, np.c_[y, x[:, 0] > 5.8, np.digitize(x[:, 2], [2, 4, 5.5])]
    # Two outputs with 2 and 3 classes:
    x, y = load_breast_cancer(return_X_y=True)
    return x, np.c_[y, np.digitize(x[:, 0], [12, 16])]


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', ['iris', 'breast_cancer'])
@pytest.mark.parametrize(
    'create',
    [
        lambda: DecisionTreeClassifier(random_state=0),
        lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
        lambda: RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=0
        ),
        lambda: ExtraTreesClassifier(
            n_estimators=10, max_depth=5, random_state=0
        ),
        lambda: KNeighborsClassifier(n_neighbors=1),
        lambda: KNeighborsClassifier(n_neighbors=5),
        lambda: KNeighborsClassifier(n_neighbors=5, weights='distance'),
    ],
    ids=[
        'DecisionTreeClassifier',
        'DecisionTreeClassifier_max_depth_3',
        'RandomForestClassifier',
        'ExtraTreesClassifier',
        'KNeighborsClassifier_k_1',
        'KNeighborsClassifier_k_5',
        'KNeighborsClassifier_k_5_distance',
    ],
)
def test_estimator_multi_output_classifier(
    request: pytest.FixtureRequest,
    tmp_root_dir: Path,
    create,
    dataset: str,
    template: str,
    language: str,
):
    """Test and compare classifiers fitted on a target with several outputs."""
    orig_est = create()

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = load_multi_output_data(dataset)
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    if isinstance(orig_est, KNeighborsClassifier):
        test_x = drop_neighbor_ties(orig_est, test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_multi_output_classifier'),
            ('estimator', request.node.callspec.id.split('-')[0]),
            ('dataset', dataset),
            ('language', language),
            ('template', template),
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
        pytest.skip('The code is too large to compile (e.g. Java attached)')
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    # The probabilities are a list of arrays, one per output:
    expected = orig_est.predict_proba(test_x)
    assert len(y_proba) == len(expected)
    for actual, desired in zip(y_proba, expected, strict=True):
        np.testing.assert_allclose(actual, desired, rtol=0, atol=1e-5)

    # A single sample:
    y_pred, y_proba = est.make(test_x[0], language=language, template=template)
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x[:1])[0])
    for actual, desired in zip(y_proba, expected, strict=True):
        np.testing.assert_allclose(actual, desired[0], rtol=0, atol=1e-5)

    assert est.test(test_x, language=language, template=template) == 1.0
