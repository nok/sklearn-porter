# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class SelectFdr(SelectorMixin):
    """Extract model data and port a SelectFdr, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.SelectFdr.html'
