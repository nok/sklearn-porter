from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.neural_network import MLPClassifier

# sklearn-porter
from sklearn_porter import Estimator, can
from tests.commons import LANGUAGES
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('n_labels', [2, 3, 5])
@pytest.mark.parametrize(
    'create',
    [
        RidgeClassifier,
        RidgeClassifierCV,
        lambda: MLPClassifier(
            hidden_layer_sizes=(8,), max_iter=500, random_state=0
        ),
    ],
    ids=['RidgeClassifier', 'RidgeClassifierCV', 'MLPClassifier'],
)
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_estimator_multilabel_classifier(
    tmp_root_dir: Path,
    create,
    n_labels: int,
    template: str,
    language: str,
):
    """Test and compare classifiers fitted on a label indicator matrix."""
    orig_est = create()

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = make_multilabel_classification(
        n_samples=200, n_features=6, n_classes=n_labels, random_state=0
    )
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_multilabel_classifier'),
            ('estimator', type(orig_est).__name__),
            ('n_labels', str(n_labels)),
            ('language', language),
            ('template', template),
        ],
    )

    y_pred, y_proba = est.make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    if can(orig_est, language, template, 'predict_proba'):
        np.testing.assert_allclose(
            y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-5
        )
    assert est.test(test_x, language=language, template=template) == 1.0
