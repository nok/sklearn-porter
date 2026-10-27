import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.ensemble import AdaBoostClassifier
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
@pytest.mark.parametrize('n_estimators', [10, 100])
@pytest.mark.parametrize('max_depth', [1, 3])
def test_estimator_ada_boost_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_estimators: int,
    max_depth: int,
):
    """Test and compare different AdaBoost classifiers."""
    orig_est = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators,
        random_state=1,
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
            ('test', 'estimator_ada_boost_classifier'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('n_estimators', str(n_estimators)),
            ('max_depth', str(max_depth)),
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
    except (exception.CodeTooLarge, exception.TooManyConstants):
        msg = (
            'Code too large for the combination: '
            f'language: {language}, template: {template}, dataset: {dataset.name}'
        )
        warnings.warn(msg, stacklevel=2)
    else:
        np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
        np.testing.assert_allclose(
            y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-5
        )
