import sys
import warnings
from pathlib import Path

import joblib


def load_model(path: Path, skip_warnings: bool = False):
    """
    Load serialized model.

    Parameters
    ----------
    path : Path
        The path to the serialized estimator.
    skip_warnings : bool (default: False)
        Avoid raised warnings.
    Returns
    -------
    The loaded BaseEstimator, pipeline or searcher.
    """
    if not path.exists() or not path.is_file():
        msg = f'File not found: {str(path)}'
        sys.exit(f'Error: {msg}')
    with warnings.catch_warnings():
        # Raise warnings like the `InconsistentVersionWarning` of scikit-learn
        # only while loading, not while porting the estimator:
        warnings.simplefilter('ignore' if skip_warnings else 'error')
        mdl = joblib.load(str(path.resolve()))
    return mdl
