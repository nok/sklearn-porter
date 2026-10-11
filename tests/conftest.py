import random as rd
import shutil
from os import environ
from pathlib import Path

import numpy as np
import pytest
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from tests.commons import RESOURCES_DIR, ROOT_DIR

# Force deterministic number generation:
np.random.seed(0)
rd.seed(0)

environ['SKLEARN_PORTER_PYTEST'] = 'True'


@pytest.fixture(scope='session')
def fitted_tree() -> DecisionTreeClassifier:
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X=x, y=y)
    return tree


@pytest.fixture(scope='session')
def joblib_model_path(
    tmp_path_factory: pytest.TempPathFactory,
    fitted_tree: DecisionTreeClassifier,
) -> Path:
    fpath = tmp_path_factory.mktemp('estimators') / 'estimator.joblib'
    dump(fitted_tree, str(fpath), compress=0)
    return fpath


@pytest.fixture(scope='session')
def tmp_root_dir(worker_id) -> Path:
    """Fixture to get the path to the temporary directory."""

    # Delete the previous generated temporary directory:
    tmp_dir = ROOT_DIR / 'tmp'
    if worker_id == 'master':
        if tmp_dir.exists():
            shutil.rmtree(str(tmp_dir.resolve()), ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Add Java dependencies:
    gson_path = RESOURCES_DIR / 'dependencies' / 'gson.jar'
    environ['SKLEARN_PORTER_PYTEST_GSON_PATH'] = str(gson_path.resolve())

    return tmp_dir
