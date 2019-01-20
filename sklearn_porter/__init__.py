# -*- coding: utf-8 -*-

from os.path import abspath
from os.path import dirname
from os.path import join
from json import load

from sklearn_porter.Porter import Porter


def _load_package_data(path):
    """Load meta data about this package from `package.json`.

    Parameters
    ----------
    path : str
        The path to the file `package.json`.

    Returns
    -------
    meta : dict
        Dictionary of key value pairs.
    """
    with open(path) as f:
        meta = load(f, encoding='utf-8')
        meta = {k: v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in meta.items()}
    return meta


file_path = abspath(dirname(__file__))
package_path = join(file_path, 'package.json')
package = _load_package_data(package_path)

__author__ = package.get('author')
__email__ = package.get('author_email')
__license__ = package.get('license')
__version__ = package.get('version')
