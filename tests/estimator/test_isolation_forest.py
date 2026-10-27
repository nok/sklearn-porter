from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.ensemble import IsolationForest

# sklearn-porter
from sklearn_porter import Estimator, can
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir


def drop_threshold_ties(est: IsolationForest, x: np.ndarray) -> np.ndarray:
    """
    Drop the samples whose score is (almost) equal to the offset. Whether they
    are inliers or outliers depends on rounding errors of the power function
    of the language and of NumPy (depending on the CPU), e.g. training samples
    are on the threshold with a float `contamination`.
    """
    return x[~np.isclose(est.decision_function(x), 0, rtol=0, atol=1e-9)]


# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'default': dict(),
    'contamination': dict(n_estimators=20, contamination=0.1),
    'max_features': dict(n_estimators=20, max_features=0.5),
    'max_samples': dict(n_estimators=20, max_samples=32),
    'bootstrap': dict(n_estimators=20, max_samples=0.5, bootstrap=True),
    'one_sample': dict(n_estimators=5, max_samples=1),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS)
def test_estimator_isolation_forest(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    params: str,
):
    """Test and compare IsolationForest outlier detectors."""
    orig_est = IsolationForest(random_state=0, **PARAMS[params])
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    x = dataset.data
    orig_est.fit(X=x)
    est = Estimator(orig_est)
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    if params != 'one_sample':
        # With a single training sample per tree all scores are exactly
        # `-2 ** -1`, which is the offset with `contamination='auto'`:
        test_x = drop_threshold_ties(orig_est, test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_isolation_forest'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('params', params),
        ],
    )

    y_pred, y_proba = est.make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
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


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
def test_isolation_forest_with_missing_values(
    tmp_root_dir: Path, language: str, template: str
):
    """Test an IsolationForest fitted on and predicting missing values."""
    rng = np.random.default_rng(0)
    x = DATASETS[1].data.copy()  # iris
    x[rng.random(x.shape) < 0.2] = np.nan
    orig_est = IsolationForest(n_estimators=20, random_state=0).fit(X=x)
    test_x = drop_threshold_ties(orig_est, x)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'isolation_forest_with_missing_values'),
            ('language', language),
            ('template', template),
        ],
    )
    y_pred, _ = Estimator(orig_est).make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
