from collections import namedtuple
from os import environ
from pathlib import Path

from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_linnerud,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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

Candidate = namedtuple('Candidate', 'name clazz create')
Dataset = namedtuple('Dataset', 'name data target')


def get_classifiers() -> list[Candidate]:
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
        MLPClassifier,
    ]
    for e in _classifiers:
        yield Candidate(name=e.__name__, clazz=e, create=e)


def get_regressors() -> list[Candidate]:
    """Get a list of available regressors."""
    _regressors = [
        DecisionTreeRegressor,
        RandomForestRegressor,
        ExtraTreesRegressor,
        MLPRegressor,
    ]
    for e in _regressors:
        yield Candidate(name=e.__name__, clazz=e, create=e)


def get_datasets() -> list[Dataset]:
    """Get a list of available toy datasets."""

    digits_ds = load_digits()
    yield Dataset(name='digits', data=digits_ds.data, target=digits_ds.target)

    iris_ds = load_iris()
    yield Dataset(name='iris', data=iris_ds.data, target=iris_ds.target)

    cancer_ds = load_breast_cancer()
    yield Dataset(
        name='breast_cancer', data=cancer_ds.data, target=cancer_ds.target
    )


def get_regression_datasets() -> list[Dataset]:
    """Get a list of available toy datasets for regressions."""

    diabetes_ds = load_diabetes()
    yield Dataset(
        name='diabetes', data=diabetes_ds.data, target=diabetes_ds.target
    )

    # Multiple outputs:
    linnerud_ds = load_linnerud()
    yield Dataset(
        name='linnerud', data=linnerud_ds.data, target=linnerud_ds.target
    )


CLASSIFIERS = list(get_classifiers())
REGRESSORS = list(get_regressors())
CANDIDATES = CLASSIFIERS + REGRESSORS
DATASETS = list(get_datasets())
REGRESSION_DATASETS = list(get_regression_datasets())
