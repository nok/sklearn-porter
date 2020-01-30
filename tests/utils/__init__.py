# -*- coding: utf-8 -*-

from os import environ, sep
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from tests.commons import (
    PORTER_N_UNI_REGRESSION_TESTS, PORTER_N_GEN_REGRESSION_TESTS
)


def fs_mkdir(base_dir: Path, dirs: List[Tuple[str, str]]) -> Path:
    sub_dirs = sep.join(['__'.join(kv) for kv in dirs])
    dir_ = base_dir / sub_dirs
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_


def dataset_generate_x(
    x: np.ndarray, n_samples: Optional[int] = None
) -> np.ndarray:
    """Helper function to create uniform test samples."""
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        msg = 'Two dimensional numpy array is required.'
        raise AssertionError(msg)
    if not n_samples:
        n_samples = PORTER_N_GEN_REGRESSION_TESTS
    return np.random.uniform(
        low=np.amin(x, axis=0),
        high=np.amax(x, axis=0),
        size=(n_samples, len(x[0])),
    )


def dataset_uniform_x(
    x: np.ndarray, n_samples: Optional[int] = None
) -> np.ndarray:
    """Helper function to pick random test samples."""
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        msg = 'Two dimensional numpy array is required.'
        raise AssertionError(msg)
    if not n_samples:
        n_samples = PORTER_N_UNI_REGRESSION_TESTS
    n_samples = min(n_samples, len(x))
    return x[(np.random.uniform(0, 1, n_samples) * (len(x) - 1)).astype(int)]
