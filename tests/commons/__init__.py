from collections.abc import Callable, Iterator
from functools import partial
from os import environ
from pathlib import Path
from typing import NamedTuple

import numpy as np
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
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Perceptron,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import (
    SVC,
    SVR,
    LinearSVC,
    LinearSVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

environ['SKLEARN_PORTER_PYTEST'] = 'True'

PORTER_N_UNI_REGRESSION_TESTS = int(
    environ.get('SKLEARN_PORTER_PYTEST_N_UNI_REGRESSION_TESTS', 15)
)
PORTER_N_GEN_REGRESSION_TESTS = int(
    environ.get('SKLEARN_PORTER_PYTEST_N_GEN_REGRESSION_TESTS', 15)
)

# The keys of all supported programming languages:
LANGUAGES = ['c', 'cs', 'go', 'java', 'js', 'php', 'ruby']

ROOT_DIR = (Path(__file__).parent / '..' / '..').resolve()
TESTS_DIR = ROOT_DIR / 'tests'
RESOURCES_DIR = ROOT_DIR / 'resources'


class Candidate(NamedTuple):
    name: str
    clazz: type
    create: Callable


# The estimators with cross-validation split the samples into 5 folds by
# default, but some tests fit them on 3 samples only:
CREATE = {
    ElasticNetCV: partial(ElasticNetCV, cv=2),
    LassoCV: partial(LassoCV, cv=2),
    LogisticRegressionCV: partial(LogisticRegressionCV, cv=2),
}


class Dataset(NamedTuple):
    name: str
    data: np.ndarray
    target: np.ndarray


def get_classifiers() -> Iterator[Candidate]:
    """Get the available classifiers."""
    _classifiers = [
        DecisionTreeClassifier,
        AdaBoostClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        LinearSVC,
        LogisticRegression,
        LogisticRegressionCV,
        SVC,
        NuSVC,
        KNeighborsClassifier,
        GaussianNB,
        BernoulliNB,
        MultinomialNB,
        ComplementNB,
        MLPClassifier,
        SGDClassifier,
        RidgeClassifier,
        RidgeClassifierCV,
        Perceptron,
    ]
    for e in _classifiers:
        yield Candidate(name=e.__name__, clazz=e, create=CREATE.get(e, e))


def get_regressors() -> Iterator[Candidate]:
    """Get the available regressors."""
    _regressors = [
        DecisionTreeRegressor,
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        MLPRegressor,
        LinearRegression,
        Ridge,
        RidgeCV,
        Lasso,
        LassoCV,
        ElasticNet,
        ElasticNetCV,
        BayesianRidge,
        ARDRegression,
        HuberRegressor,
        SGDRegressor,
        LinearSVR,
        SVR,
        NuSVR,
    ]
    for e in _regressors:
        yield Candidate(name=e.__name__, clazz=e, create=CREATE.get(e, e))


def get_outlier_detectors() -> Iterator[Candidate]:
    """Get the available outlier detectors."""
    _outlier_detectors = [
        IsolationForest,
        OneClassSVM,
    ]
    for e in _outlier_detectors:
        yield Candidate(name=e.__name__, clazz=e, create=e)


def get_datasets() -> Iterator[Dataset]:
    """Get the available toy datasets."""

    digits_ds = load_digits()
    yield Dataset(name='digits', data=digits_ds.data, target=digits_ds.target)

    iris_ds = load_iris()
    yield Dataset(name='iris', data=iris_ds.data, target=iris_ds.target)

    cancer_ds = load_breast_cancer()
    yield Dataset(
        name='breast_cancer', data=cancer_ds.data, target=cancer_ds.target
    )


def get_regression_datasets() -> Iterator[Dataset]:
    """Get the available toy datasets for regressions."""

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
OUTLIER_DETECTORS = list(get_outlier_detectors())
CANDIDATES = CLASSIFIERS + REGRESSORS + OUTLIER_DETECTORS
DATASETS = list(get_datasets())
REGRESSION_DATASETS = list(get_regression_datasets())
