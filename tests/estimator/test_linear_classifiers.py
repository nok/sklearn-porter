import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, LANGUAGES, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


ESTIMATORS = {
    'SGDClassifier-hinge': SGDClassifier(loss='hinge', random_state=1),
    'SGDClassifier-log_loss': SGDClassifier(loss='log_loss', random_state=1),
    'RidgeClassifier': RidgeClassifier(),
    'RidgeClassifierCV': RidgeClassifierCV(),
    'RidgeClassifierCV-cv': RidgeClassifierCV(cv=3, alphas=[0.01, 1.0, 100.0]),
    'Perceptron': Perceptron(random_state=1),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('name', ESTIMATORS.keys())
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_estimator_linear_classifiers(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    name: str,
    fit_intercept: bool,
):
    """Test and compare different linear classifiers."""
    orig_est = clone(ESTIMATORS[name]).set_params(fit_intercept=fit_intercept)

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    x, y = dataset.data, dataset.target
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_linear_classifiers'),
            ('estimator', name),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('fit_intercept', str(fit_intercept)),
        ],
    )

    y_pred, _ = est.make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
