import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# sklearn-porter
from sklearn_porter import Estimator, can
from tests.commons import DATASETS, LANGUAGES, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('C', [0.01, 1.0])
def test_estimator_logistic_regression(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    fit_intercept: bool,
    C: float,
):
    """Test and compare different logistic regressions."""
    orig_est = LogisticRegression(fit_intercept=fit_intercept, C=C)

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
            ('test', 'estimator_logistic_regression'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('fit_intercept', str(fit_intercept)),
            ('C', str(C)),
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
    np.testing.assert_allclose(
        y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-5
    )


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('refit', [True, False])
# The defaults of other parameters and attributes change in scikit-learn 1.10
# and 1.11, but not the coefficients:
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_estimator_logistic_regression_cv(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    fit_intercept: bool,
    refit: bool,
):
    """Test and compare logistic regressions with cross-validation."""
    # Without `refit` the coefficients are the means of the best
    # coefficients of all folds:
    orig_est = LogisticRegressionCV(
        Cs=[0.01, 1.0, 100.0], cv=3, fit_intercept=fit_intercept, refit=refit
    )

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
            ('test', 'estimator_logistic_regression_cv'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('fit_intercept', str(fit_intercept)),
            ('refit', str(refit)),
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
    np.testing.assert_allclose(
        y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-5
    )
