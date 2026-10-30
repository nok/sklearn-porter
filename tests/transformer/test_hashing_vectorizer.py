from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.feature_extraction.text import HashingVectorizer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import LANGUAGES
from tests.transformer.test_count_vectorizer import (
    DOCUMENTS,
    UNICODE_DOCUMENTS,
)
from tests.utils import fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

TEMPLATES = ['attached', 'combined', 'exported']

# Few features cause collisions of terms (also with opposite signs):
PARAMS = {
    'default': dict(n_features=2**12),
    'collisions': dict(n_features=3),
    'one_feature': dict(n_features=1, norm=None),
    'l1_without_alternate_sign': dict(
        n_features=50, norm='l1', alternate_sign=False
    ),
    'binary': dict(n_features=4, binary=True, norm=None),
    'binary_l2': dict(n_features=16, binary=True),
    'ngrams_with_stop_words': dict(
        n_features=64, ngram_range=(1, 3), stop_words='english'
    ),
    'without_lowercase': dict(n_features=64, lowercase=False),
}


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('name', PARAMS.keys())
def test_transformer_hashing_vectorizer(
    tmp_root_dir: Path,
    name: str,
    language: str,
    template: str,
):
    """Test and compare the features of a HashingVectorizer."""
    orig_est = HashingVectorizer(**PARAMS[name])
    est = Transformer(orig_est, language=language, template=template)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_hashing_vectorizer'),
            ('params', name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(DOCUMENTS, directory=tmp_dir, delete_created_files=False)
    x_true = orig_est.transform(DOCUMENTS).toarray()
    # The hashes and the sums of the signs are identical, the norms are
    # summed one after another like scikit-learn:
    np.testing.assert_array_equal(x_tr, x_true)
    assert est.test(DOCUMENTS) == 1.0


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize(
    'language', [language for language in LANGUAGES if language != 'c']
)
def test_unicode_documents(tmp_root_dir: Path, language: str, template: str):
    """The terms outside of ASCII are hashed by their UTF-8 bytes."""
    orig_est = HashingVectorizer(n_features=2**10, lowercase=False)
    est = Transformer(orig_est, language=language, template=template)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'hashing_vectorizer_unicode'),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(
        UNICODE_DOCUMENTS, directory=tmp_dir, delete_created_files=False
    )
    x_true = orig_est.transform(UNICODE_DOCUMENTS).toarray()
    np.testing.assert_array_equal(x_tr, x_true)
