# -*- coding: utf-8 -*-

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

TESTS_DIR = (Path(__file__).parent / '..').resolve()

# Parse and prepare scikit-learn version:
SKLEARN_VERSION = tuple(map(int, str(sklearn.__version__).split('.')))

Dataset = namedtuple('Dataset', 'name data target')
Candidate = namedtuple('Candidate', 'name clazz create')


def get_classifiers() -> List[Candidate]:
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
        yield (Candidate(e.__name__, e, e))


def get_regressors() -> List[Candidate]:
    _regressors = []
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPRegressor
    except ImportError:
        pass
    else:
        _regressors.append(MLPRegressor)
    for e in _regressors:
        yield (Candidate(e.__name__, e, e))


def get_datasets() -> List[Dataset]:
    _datasets = [
        Dataset('digits',
                load_digits().data,
                load_digits().target),
        Dataset('iris',
                load_iris().data,
                load_iris().target),
    ]
    try:  # for sklearn < 0.16
        from sklearn.datasets import load_breast_cancer
    except ImportError:
        pass
    else:
        _datasets.append(
            Dataset(
                'breast_cancer',
                load_breast_cancer().data,
                load_breast_cancer().target
            )
        )
    return _datasets


CLASSIFIERS = list(get_classifiers())
REGRESSORS = list(get_regressors())
CANDIDATES = CLASSIFIERS + REGRESSORS
DATASETS = list(get_datasets())
