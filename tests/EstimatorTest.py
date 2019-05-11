# -*- coding: utf-8 -*-

import pytest

import random as rd
import numpy as np

import sklearn

from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.svm.classes import LinearSVC
from sklearn.svm.classes import SVC
from sklearn.svm.classes import NuSVC
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

try:
    from sklearn.neural_network.multilayer_perceptron import MLPClassifier
except ImportError:
    MLPClassifier = None

try:
    from sklearn.neural_network.multilayer_perceptron import MLPRegressor
except ImportError:
    MLPRegressor = None

try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    GridSearchCV = None

try:
    from sklearn.model_selection import RandomizedSearchCV
except ImportError:
    RandomizedSearchCV = None

from sklearn_porter.Estimator import Estimator


# Force deterministic number generation:
np.random.seed(0)
rd.seed(0)

# Parse and prepare scikit-learn version:
SKLEARN_VERSION = tuple(map(int, str(sklearn.__version__).split('.')))


@pytest.mark.parametrize('Class', [
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
], ids=[
    'DecisionTreeClassifier',
    'AdaBoostClassifier',
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'LinearSVC',
    'SVC',
    'NuSVC',
    'KNeighborsClassifier',
    'GaussianNB',
    'BernoulliNB',
])
def test_valid_base_estimator(Class):
    """Test initialization with valid base estimator."""
    clf = Class().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    est = Estimator(clf)
    assert isinstance(est.estimator, Class)


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18),
    reason='requires scikit-learn >= v0.18'
)
@pytest.mark.parametrize('Class', [
    MLPClassifier,
    MLPRegressor
], ids=[
    'MLPClassifier',
    'MLPRegressor',
])
def test_valid_base_estimator_neural_nets(Class):
    """Test initialization with valid base estimator."""
    clf = Class().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    est = Estimator(clf)
    assert isinstance(est.estimator, Class)


@pytest.mark.parametrize('obj', [None, object, 0, 'string'])
def test_invalid_base_estimator(obj):
    """Test initialization with invalid base estimator."""
    with pytest.raises(ValueError):
        Estimator(obj)


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 15),
    reason='requires scikit-learn >= v0.15'
)
def test_extraction_from_pipeline():
    """Test the extraction of an estimator from a pipeline."""
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('SVM', SVC())])
    est = Estimator(pipeline)
    assert isinstance(est.estimator, SVC)


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 19),
    reason='requires scikit-learn >= v0.19'
)
@pytest.mark.parametrize('Class', [
    GridSearchCV,
    RandomizedSearchCV
], ids=[
    'GridSearchCV',
    'RandomizedSearchCV',
])
def test_extraction_from_optimizer(Class):
    """Test the extraction from an optimizer."""
    params = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    search = Class(SVC(gamma='scale'), params, cv=2)

    # Test unfitted optimizer:
    with pytest.raises(ValueError):
        est = Estimator(search, logger=50)
        assert isinstance(est.estimator, SVC)

    # Test fitted optimizer:
    search.fit(X=[[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
               y=[1, 2, 3, 1, 2, 3])
    est = Estimator(search, logger=50)
    assert isinstance(est.estimator, SVC)
