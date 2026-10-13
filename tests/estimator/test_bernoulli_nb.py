from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.naive_bayes import BernoulliNB

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir


def drop_class_ties(est: BernoulliNB, x: np.ndarray) -> np.ndarray:
    """
    Drop the samples whose two most likely classes have the same joint log
    likelihood. Which of them is predicted depends on rounding errors, e.g.
    two iris classes are equal after binarizing the features at the mean.
    """
    jll = np.sort(est.predict_joint_log_proba(x), axis=1)
    return x[~np.isclose(jll[:, -1], jll[:, -2])]


# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('binarize', ['zero', 'mean', 'none'])
def test_estimator_bernoulli_nb(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    binarize: str,
):
    """Test and compare BernoulliNB classifiers with different thresholds."""
    x, y = dataset.data, dataset.target
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    if binarize == 'none':
        # Without a threshold the features have to be binary already:
        threshold = np.median(x, axis=0)
        x = (x > threshold).astype(float)
        test_x = (test_x > threshold).astype(float)
    orig_est = BernoulliNB(
        binarize=dict(zero=0.0, mean=float(x.mean()), none=None)[binarize]
    )

    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)
    test_x = drop_class_ties(orig_est, test_x)
    if not len(test_x):
        # E.g. all features of iris are positive, so with `binarize=0.0` all
        # samples are equal and all classes have the same likelihood:
        pytest.skip('Skip the samples, all of them are class ties')

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_bernoulli_nb'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('binarize', binarize),
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
