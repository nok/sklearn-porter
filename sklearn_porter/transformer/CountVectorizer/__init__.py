import re
from collections.abc import Callable
from copy import copy

from loguru import logger as L

# scikit-learn
from sklearn.feature_extraction.text import (
    CountVectorizer as CountVectorizerClass,
)
from sklearn.feature_extraction.text import HashingVectorizer

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase

# The default token pattern (two or more word characters), also without the
# flag `(?u)`, which is the default of patterns of strings in Python 3:
TOKEN_PATTERNS = (r'(?u)\b\w\w+\b', r'\b\w\w+\b')


class CountVectorizer(TransformerBase, EstimatorApiABC):
    """
    Extract model data and port a CountVectorizer transformer.

    The ported transformer reads documents (see `TEXT_INPUT`) and counts the
    terms of its vocabulary: it converts the document to lowercase (with
    `lowercase=True`), splits it into tokens of two or more word characters
    (the default `token_pattern`), removes the stop words and joins the
    n-grams of the remaining tokens with spaces, like scikit-learn.
    """

    SKLEARN_URL = 'sklearn.feature_extraction.text.CountVectorizer.html'

    DEFAULT_LANGUAGE = enum.Language.JS
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.TRANSFORM

    SUPPORT = {
        language: {
            enum.Template.ATTACHED: {enum.Method.TRANSFORM},
            enum.Template.COMBINED: {enum.Method.TRANSFORM},
            enum.Template.EXPORTED: {enum.Method.TRANSFORM},
        }
        for language in enum.Language
    }

    TEXT_INPUT = True

    def __init__(self, transformer: CountVectorizerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        check_params(est, self.estimator_name)

        # `transform` of scikit-learn also accepts a vectorizer with a fixed
        # vocabulary (`vocabulary`) without a fit, which validates it:
        if not hasattr(est, 'vocabulary_'):
            est = copy(est)
            est._validate_vocabulary()
            if not est.fixed_vocabulary_:
                raise exception.NotFittedEstimatorError(self.estimator_name)
        vocabulary = sorted(est.vocabulary_, key=est.vocabulary_.__getitem__)
        n_features = len(vocabulary)
        if not n_features:
            raise exception.NotFittedEstimatorError(self.estimator_name)

        self.lowercase = bool(est.lowercase)
        self.min_n, self.max_n = map(int, est.ngram_range)
        self.binary = bool(est.binary)

        # Without n-grams only stop words of the vocabulary change the counts:
        stop_words = get_stop_words(est)
        if self.max_n == 1:
            stop_words = [w for w in stop_words if w in est.vocabulary_]
        arrays = dict(vocabulary=vocabulary)
        if stop_words:
            arrays.update(stop_words=stop_words)
        strs = tuple(arrays)
        arrays.update(self._get_weights(est, n_features))
        self._set_model_data(
            n_features, n_features_out=n_features, strs=strs, **arrays
        )
        # The order of the terms in the code of C (sorted like `strcmp`),
        # because the order of the code points is the order of UTF-8:
        self.order = sorted(range(n_features), key=vocabulary.__getitem__)

    def _get_weights(self, est: CountVectorizerClass, n_features: int) -> dict:
        """
        Get the arrays of the weights of the counts, e.g. the idf of a
        `TfidfVectorizer` (template variant `tfidf`).

        Parameters
        ----------
        est : CountVectorizerClass
            The vectorizer of scikit-learn.
        n_features : int
            The number of features (the size of the vocabulary).

        Returns
        -------
        The arrays of the weights (none of a `CountVectorizer`).
        """
        return {}

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(
            lowercase=self.lowercase,
            min_n=self.min_n,
            max_n=self.max_n,
            binary=self.binary,
            n_stop_words=len(self.model_data.get('stop_words', ())),
            order=self.order,
        )


def check_params(est: CountVectorizerClass | HashingVectorizer, name: str):
    """
    Check that the parameters of a text vectorizer are supported.

    Parameters
    ----------
    est : CountVectorizerClass or HashingVectorizer
        The text vectorizer of scikit-learn.
    name : str
        The name of the vectorizer for the error messages.

    Raises
    ------
    NotSupportedYetError
        If a parameter is not supported.
    """
    unsupported = dict(
        input=est.input != 'content',
        analyzer=est.analyzer != 'word',
        preprocessor=est.preprocessor is not None,
        tokenizer=est.tokenizer is not None,
        token_pattern=est.token_pattern not in TOKEN_PATTERNS,
        strip_accents=bool(est.strip_accents),
        ngram_range=est.ngram_range[0] < 1,
    )
    for param, is_unsupported in unsupported.items():
        if is_unsupported:
            msg = (
                f'The parameter `{param}={getattr(est, param)!r}` of the '
                f'transformer `{name}` is not supported yet.'
            )
            raise exception.NotSupportedYetError(msg)


def get_stop_words(est: CountVectorizerClass | HashingVectorizer) -> list[str]:
    """
    Get the stop words of a text vectorizer which can be tokens.

    Only stop words of two or more word characters (in lowercase with
    `lowercase=True`) can be removed from the tokens.

    Parameters
    ----------
    est : CountVectorizerClass or HashingVectorizer
        The text vectorizer of scikit-learn.

    Returns
    -------
    The sorted stop words.
    """
    return sorted(
        word
        for word in est.get_stop_words() or ()
        if re.fullmatch(r'\w\w+', word)
        and (not est.lowercase or word == word.lower())
    )
