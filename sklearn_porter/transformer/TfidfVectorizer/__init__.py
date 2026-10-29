from collections.abc import Callable

import numpy as np

# scikit-learn
from sklearn.feature_extraction.text import (
    CountVectorizer as CountVectorizerClass,
)
from sklearn.feature_extraction.text import (
    TfidfVectorizer as TfidfVectorizerClass,
)

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.transformer.CountVectorizer import CountVectorizer


class TfidfVectorizer(CountVectorizer):
    """
    Extract model data and port a TfidfVectorizer transformer.

    The ported transformer counts the terms like a `CountVectorizer` and
    weights the counts like a `TfidfTransformer` (with the templates of the
    `CountVectorizer` and the template variable `tfidf`).
    """

    SKLEARN_URL = 'sklearn.feature_extraction.text.TfidfVectorizer.html'

    def __init__(self, transformer: TfidfVectorizerClass):
        # Unlike a `CountVectorizer` a `TfidfVectorizer` needs a fit, also
        # with a fixed vocabulary:
        if not hasattr(transformer, '_tfidf'):
            name = type(transformer).__name__
            raise exception.NotFittedEstimatorError(name)
        super().__init__(transformer)

    def _get_weights(self, est: CountVectorizerClass, n_features: int) -> dict:
        # `transform` of scikit-learn uses the `TfidfTransformer` of the fit
        # (`_tfidf`) with its parameters, not the current parameters of the
        # vectorizer. It weights the counts with `idf_`, if the fit created
        # it (`use_idf=True`); multiplying by one doesn't change the counts:
        tfidf = est._tfidf  # ty: ignore[unresolved-attribute]
        self.norm = tfidf.norm
        self.sublinear_tf = bool(tfidf.sublinear_tf)
        idf = getattr(tfidf, 'idf_', None)
        if idf is None:
            idf = np.ones(n_features)
        return dict(idf=idf)

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(
            super().literals(converter),
            tfidf=True,
            norm=self.norm,
            sublinear_tf=self.sublinear_tf,
        )
