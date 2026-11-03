from logging import NullHandler, getLogger

from sklearn_porter import meta
from sklearn_porter.Estimator import Estimator
from sklearn_porter.Pipeline import Pipeline
from sklearn_porter.Porter import Porter, can, make, port, save, show, test
from sklearn_porter.Transformer import Transformer

__all__ = [
    'Estimator',
    'Pipeline',
    'Porter',
    'Transformer',
    'can',
    'make',
    'port',
    'save',
    'show',
    'test',
]

# The application decides whether and where the debug messages appear:
getLogger(__name__).addHandler(NullHandler())

__author__ = meta.__author__
__email__ = meta.__email__
__license__ = meta.__license__
__version__ = meta.__version__
