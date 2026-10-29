from collections.abc import Callable
from math import log

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import PowerTransformer as PowerTransformerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase

# The machine epsilon, which SciPy uses to choose the formula of the
# Yeo-Johnson transformation (`scipy.stats.yeojohnson`):
EPS = float(np.finfo(np.float64).eps)


class PowerTransformer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a PowerTransformer transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.PowerTransformer.html'

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

    def __init__(self, transformer: PowerTransformerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # scikit-learn transforms each feature with `scipy.stats.yeojohnson`
        # or `scipy.special.boxcox` and the parameter `lambdas_[i]`, then it
        # standardizes the features with a `StandardScaler` (only an
        # internal attribute, without `standardize=True` the ported
        # transformer subtracts zero and divides by one):
        self.method = est.method
        lambdas = np.asarray(est.lambdas_, dtype=np.float64)
        mean, scale = np.zeros(n_features), np.ones(n_features)
        if est.standardize:
            scaler = est._scaler
            mean, scale = scaler.mean_, scaler.scale_
        self.lambdas = lambdas.tolist()
        self._set_model_data(
            n_features, lambdas=lambdas, mean=mean, scale=scale
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        # The template `combined` gets the formula of each feature and the
        # constant parts of the formulas as numbers (Go computes constant
        # expressions exactly, not with floating-point numbers):
        lambdas = self.lambdas
        return dict(
            method=self.method,
            eps=converter(EPS),
            pos_log=[abs(lam) < EPS for lam in lambdas],
            neg_log=[not abs(lam - 2) > EPS for lam in lambdas],
            bc_log=[abs(lam) < 1e-19 for lam in lambdas],
            two_minus=[converter(2 - lam) for lam in lambdas],
            reciprocals=[converter(1 / lam if lam else 0.0) for lam in lambdas],
            signs=[converter(1.0 if lam > 0 else -1.0) for lam in lambdas],
            log_abs=[
                converter(log(abs(lam)) if lam else 0.0) for lam in lambdas
            ],
        )
