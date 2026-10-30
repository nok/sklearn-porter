from collections.abc import Callable

from loguru import logger as L

# scikit-learn
from sklearn.feature_extraction.text import (
    HashingVectorizer as HashingVectorizerClass,
)

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.CountVectorizer import (
    check_params,
    get_stop_words,
)
from sklearn_porter.transformer.TransformerBase import TransformerBase


class HashingVectorizer(TransformerBase, EstimatorApiABC):
    """
    Extract model data and port a HashingVectorizer transformer.

    The ported transformer reads documents (see `TEXT_INPUT`) and splits
    them into terms like a `CountVectorizer`, but instead of a vocabulary it
    maps each term to a feature with the signed 32-bit MurmurHash3 (x86_32,
    seed 0) of its UTF-8 bytes: the index is `abs(hash) % n_features`, the
    sign (with `alternate_sign=True`) the sign of the hash, like scikit-learn.
    Finally the features are normalized (`norm`). The model data consists of
    the stop words only, so the ported transformer is small.
    """

    SKLEARN_URL = 'sklearn.feature_extraction.text.HashingVectorizer.html'

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

    def __init__(self, transformer: HashingVectorizerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        # The vectorizer is stateless, scikit-learn only validates the
        # parameters in `fit`:
        est._validate_params()
        check_params(est, self.estimator_name)

        n_features = int(est.n_features)
        self.lowercase = bool(est.lowercase)
        self.min_n, self.max_n = map(int, est.ngram_range)
        self.binary = bool(est.binary)
        self.alternate_sign = bool(est.alternate_sign)
        self.norm = est.norm

        stop_words = get_stop_words(est)
        arrays = dict(stop_words=stop_words) if stop_words else {}
        self._set_model_data(
            n_features, n_features_out=n_features, strs=arrays, **arrays
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(
            lowercase=self.lowercase,
            min_n=self.min_n,
            max_n=self.max_n,
            binary=self.binary,
            alternate_sign=self.alternate_sign,
            norm=self.norm,
            n_stop_words=len(self.model_data.get('stop_words', ())),
        )
