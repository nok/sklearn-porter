# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class SelectKBest(SelectorMixin):
    """Extract model data and port a SelectKBest, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.SelectKBest.html'
