# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class SequentialFeatureSelector(SelectorMixin):
    """Extract model data and port a SequentialFeatureSelector, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.SequentialFeatureSelector.html'
