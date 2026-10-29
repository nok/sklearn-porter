# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class SelectFwe(SelectorMixin):
    """Extract model data and port a SelectFwe, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.SelectFwe.html'
