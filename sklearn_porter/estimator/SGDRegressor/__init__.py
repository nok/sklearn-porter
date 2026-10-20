# scikit-learn
from sklearn.linear_model import SGDRegressor as SGDRegressorClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class SGDRegressor(LinearRegression, EstimatorBase):
    """Extract model data and port a SGDRegressor regressor."""

    SKLEARN_URL = 'sklearn.linear_model.SGDRegressor.html'

    def __init__(self, estimator: SGDRegressorClass):
        super().__init__(estimator)
        # `sparsify()` converts the coefficients to a sparse matrix with one
        # row, which becomes a 2D array, but the regressor always has a
        # single output:
        self.meta_info['is_multi_output'] = False
