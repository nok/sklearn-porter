from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import LANGUAGES
from tests.transformer.test_count_vectorizer import CORPUS, DOCUMENTS
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
    'binary': dict(binary=True),
    'ngrams_with_stop_words': dict(ngram_range=(1, 2), stop_words='english'),
    'fixed_vocabulary': dict(vocabulary=['the', 'document', 'the first']),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('name', PARAMS.keys())
def test_transformer_tfidf_vectorizer(
    tmp_root_dir: Path,
    name: str,
    language: str,
    template: str,
):
    """Test and compare the tf-idf features of a TfidfVectorizer."""
    orig_est = TfidfVectorizer(**PARAMS[name]).fit(CORPUS)
    est = Transformer(orig_est, language=language, template=template)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_tfidf_vectorizer'),
            ('params', name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(DOCUMENTS, directory=tmp_dir, delete_created_files=False)
    x_true = orig_est.transform(DOCUMENTS).toarray()
    if orig_est.sublinear_tf:
        # NumPy computes the logarithm with SIMD instructions depending on
        # the CPU, so the results can differ by a few ULP:
        np.testing.assert_allclose(x_tr, x_true, rtol=1e-14, atol=0)
    else:
        # The ported vectorizer sums the weighted counts one after another
        # like scikit-learn (the sorted non-zero values of a sparse row):
        np.testing.assert_array_equal(x_tr, x_true)
    assert est.test(DOCUMENTS) == 1.0
