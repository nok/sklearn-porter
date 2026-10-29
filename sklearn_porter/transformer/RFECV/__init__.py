# sklearn-porter
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class RFECV(SelectorMixin):
    """Extract model data and port an RFECV, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.RFECV.html'
