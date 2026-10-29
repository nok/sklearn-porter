# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class SelectPercentile(SelectorMixin):
    """Extract model data and port a SelectPercentile, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.SelectPercentile.html'
