# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class GenericUnivariateSelect(SelectorMixin):
    """Extract model data and port a GenericUnivariateSelect, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.GenericUnivariateSelect.html'
