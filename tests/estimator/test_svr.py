from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, NuSVR, OneClassSVM

# sklearn-porter
from sklearn_porter import Estimator, can
from sklearn_porter import exceptions as exception
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir


def make(est, x: np.ndarray, tmp_dir: Path, language: str, template: str):
    try:
        return Estimator(est).make(
            x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except (exception.CodeTooLarge, exception.TooManyConstants) as e:
        pytest.skip(str(e))


# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

KERNELS = {
    'linear': dict(kernel='linear'),
    'poly': dict(kernel='poly'),
    'poly-2': dict(kernel='poly', degree=2, coef0=1.0),
    'rbf': dict(kernel='rbf'),
    'rbf-0_01': dict(kernel='rbf', gamma=0.01),
    'sigmoid': dict(kernel='sigmoid', gamma='auto', coef0=0.5),
}

REGRESSORS = {
    'SVR': lambda **kw: SVR(C=10.0, **kw),
    'SVR-epsilon': lambda **kw: SVR(C=10.0, epsilon=0.5, **kw),
    'NuSVR': lambda **kw: NuSVR(C=10.0, **kw),
    'NuSVR-nu': lambda **kw: NuSVR(C=10.0, nu=0.1, **kw),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('dataset', REGRESSION_DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('name', REGRESSORS)
def test_estimator_svr(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    kernel: str,
    name: str,
):
    """Test and compare different SVM regressors."""
    # Limit the iterations, because libsvm may not converge (e.g. `poly`):
    orig_est = REGRESSORS[name](max_iter=100_000, **KERNELS[kernel])

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    # Scale the features and the targets, otherwise libsvm takes extremely
    # long to converge. The SVMs predict a single output:
    x = StandardScaler().fit_transform(dataset.data)
    y = np.asarray(dataset.target, dtype=float)
    if y.ndim > 1:
        y = y[:, 0]
    y = (y - y.mean()) / y.std()
    orig_est.fit(X=x, y=y)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_svr'),
            ('estimator', name),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('kernel', kernel),
        ],
    )

    y_pred, _ = make(orig_est, test_x, tmp_dir, language, template)
    np.testing.assert_allclose(
        y_pred.astype(float), orig_est.predict(test_x), rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
def test_svr_without_support_vectors(
    tmp_root_dir: Path, template: str, language: str
):
    """Test a SVR without support vectors, which predicts the intercept."""
    orig_est = SVR(epsilon=1000.0)
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = REGRESSION_DATASETS[0].data, REGRESSION_DATASETS[0].target
    orig_est.fit(X=x, y=y)
    assert len(orig_est.support_vectors_) == 0
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'svr_without_support_vectors'),
            ('language', language),
            ('template', template),
        ],
    )
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    y_pred, _ = Estimator(orig_est).make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(
        y_pred.astype(float), orig_est.predict(test_x)
    )


def drop_boundary_samples(est: OneClassSVM, x: np.ndarray) -> np.ndarray:
    """
    Drop the samples whose decision value is (almost) 0. Whether they are
    inliers or outliers depends on rounding errors, because libsvm computes
    the dot products with BLAS.
    """
    return x[~np.isclose(est.decision_function(x), 0, rtol=0, atol=1e-9)]


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('nu', [0.1, 0.5])
def test_estimator_one_class_svm(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    kernel: str,
    nu: float,
):
    """Test and compare OneClassSVM outlier detectors."""
    orig_est = OneClassSVM(nu=nu, max_iter=100_000, **KERNELS[kernel])
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    x = StandardScaler().fit_transform(dataset.data)
    orig_est.fit(X=x)
    est = Estimator(orig_est)
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    test_x = drop_boundary_samples(orig_est, test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_one_class_svm'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('kernel', kernel),
            ('nu', str(nu).replace('.', '_')),
        ],
    )

    y_pred, y_proba = make(orig_est, test_x, tmp_dir, language, template)
    assert y_proba is None
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    score = est.test(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    assert score == 1.0
