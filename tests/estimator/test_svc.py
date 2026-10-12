import warnings
from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC

from sklearn_porter import exceptions as exception

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
@pytest.mark.parametrize(
    'gamma', [0.001, 0.01, 'auto'], ids=['0_001', '0_01', 'auto']
)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
def test_estimator_svc(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    kernel: str,
    gamma: str | float,
):
    """Test and compare different SVM classifiers."""
    # Limit the iterations, because libsvm may not converge (e.g. `poly`):
    orig_est = SVC(kernel=kernel, gamma=gamma, max_iter=100_000, random_state=1)

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    # Scale the features, otherwise libsvm takes extremely long to converge:
    x, y = StandardScaler().fit_transform(dataset.data), dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    if not isinstance(gamma, str):
        gamma = str(gamma).replace('.', '_')
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_svc'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('kernel', kernel),
            ('gamma', gamma),
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
    except exception.NotSupportedYetError as e:
        if 'template' == 'precomputed':
            msg = 'The passed kernel `precomputed` is not supported.'
            assert msg in str(e)
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


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
def test_estimator_nu_svc(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    kernel: str,
):
    """Test and compare different NuSVC classifiers."""
    # Limit the iterations, because libsvm may not converge (e.g. `poly`):
    orig_est = NuSVC(kernel=kernel, max_iter=100_000, random_state=1)

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    # Scale the features, otherwise libsvm takes extremely long to converge:
    x, y = StandardScaler().fit_transform(dataset.data), dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_nu_svc'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('kernel', kernel),
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
    else:
        assert score == 1.0
