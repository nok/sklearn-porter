from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import Estimator, can
from sklearn_porter import exceptions as exception
from tests.commons import DATASETS, LANGUAGES, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('weights', ['sample_weight', 'class_weight'])
@pytest.mark.parametrize(
    'create',
    [
        lambda **kw: DecisionTreeClassifier(max_depth=4, random_state=0, **kw),
        lambda **kw: RandomForestClassifier(
            n_estimators=10, max_depth=4, random_state=0, **kw
        ),
        lambda **kw: ExtraTreesClassifier(
            n_estimators=10, max_depth=4, random_state=0, **kw
        ),
    ],
    ids=[
        'DecisionTreeClassifier',
        'RandomForestClassifier',
        'ExtraTreesClassifier',
    ],
)
def test_estimator_weighted_tree_classifier(
    request: pytest.FixtureRequest,
    tmp_root_dir: Path,
    create,
    weights: str,
    dataset: Dataset,
    template: str,
    language: str,
):
    """Test and compare tree classifiers fitted on weighted samples."""
    x, y = dataset.data, dataset.target

    # The weights change the class fractions of the leaves:
    sample_weight = None
    if weights == 'sample_weight':
        orig_est = create()
        sample_weight = np.random.RandomState(0).uniform(0.1, 3.0, len(y))
    else:
        classes = np.unique(y)
        class_weight = np.linspace(0.5, 3.0, len(classes))
        orig_est = create(
            class_weight=dict(zip(classes, class_weight, strict=True))
        )

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    orig_est.fit(X=x, y=y, sample_weight=sample_weight)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_weighted_tree_classifier'),
            ('estimator', request.node.callspec.id.split('-')[0]),
            ('weights', weights),
            ('dataset', dataset.name),
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
    # The ported trees return the class fractions of the leaves and the
    # forests average them like scikit-learn (only additions and a division,
    # no functions like `exp`), so the probabilities are identical:
    np.testing.assert_array_equal(y_proba, orig_est.predict_proba(test_x))
