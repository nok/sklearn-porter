import random as rd
import urllib.request
from os import environ, getpid, pathsep
from pathlib import Path

import numpy as np
import pytest
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_porter.language.java import Java

# Force deterministic number generation:
np.random.seed(0)
rd.seed(0)

environ['SKLEARN_PORTER_PYTEST'] = 'True'


@pytest.fixture(scope='session', autouse=True)
def strict_php(tmp_path_factory: pytest.TempPathFactory):
    """
    Fixture to show all PHP errors on stdout.

    The default `php.ini` of the Docker image hides deprecations. With this
    configuration notices, warnings and deprecations (e.g. dynamic properties
    since PHP 8.2) break the JSON output of the ported estimators, so that
    the regression tests fail.
    """
    ini_dir = tmp_path_factory.mktemp('php')
    ini_path = ini_dir / 'strict.ini'
    ini_path.write_text('error_reporting = E_ALL\ndisplay_errors = On\n')
    # The leading separator keeps the default scan directory:
    environ['PHP_INI_SCAN_DIR'] = pathsep + str(ini_dir)


@pytest.fixture(scope='session', autouse=True)
def single_thread():
    """
    Fixture to use a single thread of OpenMP and BLAS per test process.

    The HistGradientBoosting estimators (OpenMP) and the linear algebra
    (BLAS) use threads on all CPUs, so the parallel test processes of
    pytest-xdist would slow each other down.
    """
    with threadpool_limits(limits=1):
        yield


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
    dump(fitted_tree, str(fpath))
    return fpath


@pytest.fixture(scope='session')
def tmp_root_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Fixture to get the path to the temporary directory.

    The directory is not in the repository, because in Docker the repository
    is a mounted volume (e.g. sshfs with Colima) on which freshly compiled
    binaries are not always executable immediately.
    """
    return tmp_path_factory.mktemp('tmp')


@pytest.fixture(scope='session')
def gson_path(request: pytest.FixtureRequest) -> Path:
    """
    Fixture to get the path to GSON, the dependency of the Java templates
    `exported`.

    The Docker image sets `SKLEARN_PORTER_PYTEST_GSON_PATH`, otherwise the
    jar is downloaded once to the cache of pytest (`.pytest_cache`).
    """
    if environ.get('SKLEARN_PORTER_PYTEST_GSON_PATH'):
        return Path(environ['SKLEARN_PORTER_PYTEST_GSON_PATH'])
    path = request.config.cache.mkdir('gson') / 'gson.jar'
    if not path.exists():
        # A temporary file per process, so that the parallel test processes
        # of pytest-xdist don't read an incomplete jar:
        tmp_path = path.with_name(f'gson.{getpid()}.tmp')
        urllib.request.urlretrieve(Java.GSON_DOWNLOAD_URI, str(tmp_path))
        tmp_path.replace(path)
    environ['SKLEARN_PORTER_PYTEST_GSON_PATH'] = str(path)
    return path


@pytest.fixture(autouse=True)
def java_dependencies(request: pytest.FixtureRequest):
    """Fixture to provide GSON for the regression tests only."""
    if request.node.get_closest_marker('regression'):
        request.getfixturevalue('gson_path')
