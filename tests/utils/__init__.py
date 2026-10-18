from os import sep
from pathlib import Path

import numpy as np

from tests.commons import (
    PORTER_N_GEN_REGRESSION_TESTS,
    PORTER_N_UNI_REGRESSION_TESTS,
)


def fs_mkdir(base_dir: Path, dirs: list[tuple[str, str]]) -> Path:
    sub_dirs = sep.join(['__'.join(kv) for kv in dirs])
    dir_ = base_dir / sub_dirs
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_


def dataset_generate_x(
    x: np.ndarray, n_samples: int | None = None
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
    x: np.ndarray, n_samples: int | None = None
) -> np.ndarray:
    """Helper function to pick random test samples."""
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        msg = 'Two dimensional numpy array is required.'
        raise AssertionError(msg)
    if not n_samples:
        n_samples = PORTER_N_UNI_REGRESSION_TESTS
    n_samples = min(n_samples, len(x))
    return x[(np.random.uniform(0, 1, n_samples) * (len(x) - 1)).astype(int)]


def drop_neighbor_ties(est, x: np.ndarray) -> np.ndarray:
    """
    Drop the samples whose k-th and (k+1)-th nearest neighbors have the same
    distance. Which of them is used depends on the ordering of the training
    data and the sorting algorithm, so the results of the ported estimator
    and scikit-learn can differ for them.
    """
    k = est.n_neighbors
    n_samples = len(est._fit_X)
    if k >= n_samples:
        return x
    dists, _ = est.kneighbors(x, n_neighbors=k + 1)
    return x[~np.isclose(dists[:, k - 1], dists[:, k])]


def drop_class_ties(est, x: np.ndarray) -> np.ndarray:
    """
    Drop the samples whose two most likely classes of a naive Bayes
    classifier have the same joint log likelihood. Which of them is predicted
    depends on rounding errors, e.g. two iris classes of a BernoulliNB are
    equal after binarizing the features at the mean.
    """
    jll = est.predict_joint_log_proba(x)
    if jll.shape[1] < 2:
        return x
    jll = np.sort(jll, axis=1)
    return x[~np.isclose(jll[:, -1], jll[:, -2])]
