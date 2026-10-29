# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class VarianceThreshold(SelectorMixin):
    """Extract model data and port a VarianceThreshold, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.VarianceThreshold.html'
