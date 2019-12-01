# -*- coding: utf-8 -*-

import sys
import warnings
import joblib

from pathlib import Path


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
        msg = 'File not found: {}'.format(str(path))
        sys.exit('Error: {}'.format(msg))
    if skip_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mdl = joblib.load(str(path.resolve()))
    else:
        warnings.filterwarnings('error')
        mdl = joblib.load(str(path.resolve()))
    return mdl
