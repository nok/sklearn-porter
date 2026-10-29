# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class RFE(SelectorMixin):
    """Extract model data and port an RFE, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.RFE.html'
