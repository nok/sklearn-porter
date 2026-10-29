from importlib import import_module
from pathlib import Path
from subprocess import CompletedProcess, run

import numpy as np
import pytest

# scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import LANGUAGES
from tests.utils import fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

BASE_MODULE = import_module('sklearn_porter.Base')
TEMPLATES = ['attached', 'combined', 'exported']

CORPUS = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    "Don't stop_me now: it's 2024, OK? A b c d_e x1 42 _",
    'UPPER lower MiXeD Case CASE case',
    'the the the and and of of to',
]

# Documents with punctuation, digits, underscores, single characters, line
# breaks, tabs and null characters, empty documents and unknown words:
DOCUMENTS = CORPUS + [
    '',
    ' ',
    'a',
    'unknown words only',
    'first\nsecond\r\nthird\0fourth',
    'tab\tseparated\x0bvertical\x0cform-feed',
    'document document DOCUMENT',
    'is the is the is the',
    'xxx ' + 'first ' * 50,
    '(this)[is]{the}<first>"document"',
    b'the first document in bytes',
]

PARAMS = {
    'default': dict(),
    'ngrams': dict(ngram_range=(1, 3)),
    'bigrams': dict(ngram_range=(2, 2)),
    'stop_words': dict(stop_words='english', ngram_range=(1, 2)),
    'binary': dict(binary=True),
    'without_lowercase': dict(lowercase=False),
    'fixed_vocabulary': dict(
        vocabulary=['the', 'document', 'the first', 'xyz', 'is', 'OK']
    ),
    'fixed_vocabulary_with_stop_words': dict(
        vocabulary=['the', 'document', 'is', 'this'], stop_words=['the', 'is']
    ),
}

# Documents with characters outside of ASCII:
UNICODE_DOCUMENTS = [
    'Größe der STRAßE, Übermäßig ÄRGER über Öl',
    "L'été à Noël: ÉCOLE, Ça va? naïve café",
    'El niño ÑANDÚ come piña. ¿Qué?',
    'Привет МИР, ёлка и ЁЖИК',
    '東京 タワー 한국어 中文字',
    'नमस्ते दुनिया हिंदी',
    'happy 😀 days 👍🏽 ok',
    '“quoted” — dash… don’t ‘x’ «a» bb',
    'x² ½ ２０２４ ①② ⅫⅢ ٣٤',
    'café naïve',
]


def _vectorizer(params: dict) -> CountVectorizer:
    orig_est = CountVectorizer(**params)
    if 'vocabulary' not in params:  # without a fit
        orig_est.fit(CORPUS)
    return orig_est


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('name', PARAMS.keys())
def test_transformer_count_vectorizer(
    tmp_root_dir: Path,
    name: str,
    language: str,
    template: str,
):
    """Test and compare the counts of a CountVectorizer."""
    orig_est = _vectorizer(PARAMS[name])
    est = Transformer(orig_est, language=language, template=template)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_count_vectorizer'),
            ('params', name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(DOCUMENTS, directory=tmp_dir, delete_created_files=False)
    x_true = orig_est.transform(DOCUMENTS).toarray()
    np.testing.assert_array_equal(x_tr, x_true)
    # A single document:
    np.testing.assert_array_equal(est.make(DOCUMENTS[1]), x_true[1])
    assert est.test(DOCUMENTS) == 1.0


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_documents_of_arguments(
    tmp_root_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    template: str,
):
    """The ported vectorizer joins the arguments to one document."""
    cmds = []

    def run_and_record(cmd: list[str], **kwargs) -> CompletedProcess:
        cmds.append(cmd)
        return run(cmd, **kwargs)

    monkeypatch.setattr(BASE_MODULE, 'run', run_and_record)

    orig_est = _vectorizer(PARAMS['ngrams'])
    est = Transformer(orig_est, language=language, template=template)
    tmp_dir = tmp_root_dir / f'count_vectorizer_arguments-{language}-{template}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    est.make(CORPUS[:1], directory=tmp_dir, delete_created_files=False)
    args = ['Is', 'this the', 'FIRST', 'document?']
    proc = run(
        cmds[0] + args, capture_output=True, encoding='utf-8', check=True
    )
    values = [float(v) for v in proc.stdout.split()]
    x_true = orig_est.transform([' '.join(args)]).toarray()[0]
    np.testing.assert_array_equal(values, x_true)


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize(
    'language', [language for language in LANGUAGES if language != 'c']
)
def test_unicode_documents(tmp_root_dir: Path, language: str, template: str):
    """
    Without the conversion to lowercase the tokens of documents with
    characters outside of ASCII are identical (except in C).
    """
    orig_est = CountVectorizer(lowercase=False).fit(UNICODE_DOCUMENTS)
    est = Transformer(orig_est, language=language, template=template)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'count_vectorizer_unicode'),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(
        UNICODE_DOCUMENTS, directory=tmp_dir, delete_created_files=False
    )
    x_true = orig_est.transform(UNICODE_DOCUMENTS).toarray()
    np.testing.assert_array_equal(x_tr, x_true)
