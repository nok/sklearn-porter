from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.feature_extraction.text import TfidfTransformer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import LANGUAGES
from tests.utils import fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

PARAMS = {
    'default': dict(),
    'l1': dict(norm='l1'),
    'without_norm': dict(norm=None),
    'without_idf': dict(use_idf=False),
    'without_smoothing': dict(smooth_idf=False),
    'sublinear_tf': dict(sublinear_tf=True),
    'sublinear_tf_l1_without_idf': dict(
        sublinear_tf=True, norm='l1', use_idf=False
    ),
    'sublinear_tf_without_norm': dict(sublinear_tf=True, norm=None),
}


def _counts(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Counts of terms (with different frequencies) for the fit and samples
    with zeros, negative zeros, fractions and norms which underflow or
    overflow.
    """
    rng = np.random.default_rng(0)
    rates = rng.uniform(0.05, 3.0, size=30)
    x = rng.poisson(rates, size=(200, 30)).astype(float)
    test_x = np.vstack(
        (
            rng.poisson(2 * rates, size=(300, 30)).astype(float),
            rng.uniform(0.0, 50.0, size=(20, 30)),
            np.zeros((1, 30)),
            np.full((1, 30), -0.0),
            np.full((1, 30), 1e-200),
            np.full((1, 30), 1e300),
        )
    )
    # scikit-learn raises an error for the missing values of the logarithm
    # of negative counts, if it normalizes them:
    if not params.get('sublinear_tf') or params.get('norm', 'l2') is None:
        test_x = np.vstack((test_x, -rng.poisson(rates, size=(20, 30))))
    return x, test_x


# scikit-learn computes the logarithm of negative counts:
@pytest.mark.filterwarnings('ignore:invalid value encountered in log')
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_tfidf_transformer(
    tmp_root_dir: Path,
    params: dict,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a TfidfTransformer."""
    x, test_x = _counts(params)
    orig_est = TfidfTransformer(**params).fit(x)
    est = Transformer(orig_est, language=language, template=template)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_tfidf_transformer'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    x_true = orig_est.transform(test_x).toarray()
    if orig_est.sublinear_tf:
        # NumPy computes the logarithm with SIMD instructions depending on
        # the CPU, so the results can differ by a few ULP:
        np.testing.assert_allclose(x_tr, x_true, rtol=1e-14, atol=0)
    else:
        # The ported transformer sums the features one after another like
        # scikit-learn (the non-zero values of a sparse row):
        np.testing.assert_array_equal(x_tr, x_true)
    is_nan = np.isnan(x_true)
    np.testing.assert_array_equal(
        np.signbit(x_tr[~is_nan]), np.signbit(x_true[~is_nan])
    )
    assert est.test(test_x) == 1.0
