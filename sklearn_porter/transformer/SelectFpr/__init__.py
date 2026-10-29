# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class SelectFpr(SelectorMixin):
    """Extract model data and port a SelectFpr, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.SelectFpr.html'
