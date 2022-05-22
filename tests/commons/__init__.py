from collections import namedtuple
from os import environ
from pathlib import Path
from typing import List

from sklearn.ensemble.forest import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree.tree import DecisionTreeClassifier

import sklearn
from sklearn.datasets import load_digits, load_iris

environ['SKLEARN_PORTER_PYTEST'] = 'True'

PORTER_N_UNI_REGRESSION_TESTS = environ.get(
    'SKLEARN_PORTER_PYTEST_N_UNI_REGRESSION_TESTS', 15
)
PORTER_N_GEN_REGRESSION_TESTS = environ.get(
    'SKLEARN_PORTER_PYTEST_N_GEN_REGRESSION_TESTS', 15
)

ROOT_DIR = (Path(__file__).parent / '..' / '..').resolve()
TESTS_DIR = ROOT_DIR / 'tests'
RESOURCES_DIR = ROOT_DIR / 'resources'

# Parse and prepare scikit-learn version:
SKLEARN_VERSION = tuple(map(int, str(sklearn.__version__).split('.')))

Candidate = namedtuple('Candidate', 'name clazz create')
Dataset = namedtuple('Dataset', 'name data target')


def get_classifiers() -> List[Candidate]:
    """Get a list of available classifiers."""
    _classifiers = [
        DecisionTreeClassifier,
        AdaBoostClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
        LinearSVC,
        SVC,
        NuSVC,
        KNeighborsClassifier,
        GaussianNB,
        BernoulliNB,
    ]
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPClassifier
    except ImportError:
        pass
    else:
        _classifiers.append(MLPClassifier)
    for e in _classifiers:
        yield Candidate(name=e.__name__, clazz=e, create=e)


def get_regressors() -> List[Candidate]:
    """Get a list of available regressors."""
    _regressors = []
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPRegressor
    except ImportError:
        pass
    else:
        _regressors.append(MLPRegressor)
    for e in _regressors:
        yield Candidate(name=e.__name__, clazz=e, create=e)


def get_datasets() -> List[Dataset]:
    """Get a list of available toy datasets."""

    digits_ds = load_digits()
    yield Dataset(name='digits', data=digits_ds.data, target=digits_ds.target)

    iris_ds = load_iris()
    yield Dataset(name='iris', data=iris_ds.data, target=iris_ds.target)

    try:  # for sklearn < 0.16
        from sklearn.datasets import load_breast_cancer
    except ImportError:
        pass
    else:
        cancer_ds = load_breast_cancer()
        yield Dataset(
            name='breast_cancer', data=cancer_ds.data, target=cancer_ds.target
        )


CLASSIFIERS = list(get_classifiers())
REGRESSORS = list(get_regressors())
CANDIDATES = CLASSIFIERS + REGRESSORS
DATASETS = list(get_datasets())
