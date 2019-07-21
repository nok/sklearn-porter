# -*- coding: utf-8 -*-

from sys import version_info as PYTHON_VERSION
from os import environ
from typing import Tuple, Dict, Optional, Callable
from pathlib import Path
import shutil
import urllib.request

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

from sklearn_porter.language.Java import Java
from sklearn_porter.exceptions import NotFittedEstimatorError,\
    InvalidLanguageError, InvalidTemplateError

from sklearn.datasets import load_digits, load_iris
try:  # for sklearn < 0.16
    from sklearn.datasets import load_breast_cancer
except ImportError:
    load_breast_cancer = lambda: None

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

# Check python version:
if PYTHON_VERSION[:2] < (3, 5):
    pytest.skip('tests requires python >= 3.5', allow_module_level=True)

# Parse and prepare scikit-learn version:
SKLEARN_VERSION = tuple(map(int, str(sklearn.__version__).split('.')))

environ['SKLEARN_PORTER_PYTEST'] = 'True'


@pytest.fixture(scope='session')
def tmp(worker_id) -> Path:
    """Fixture to get the path to the temporary directory."""

    tmp = Path(__file__).parent / 'tmp'
    if worker_id is 'master':
        if tmp.exists():
            shutil.rmtree(str(tmp.resolve()), ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # Download GSON:
    url = Java.GSON_DOWNLOAD_URI
    path = tmp / 'gson.jar'
    if not path.exists():
        urllib.request.urlretrieve(url, str(path))
    environ['SKLEARN_PORTER_PYTEST_GSON_PATH'] = str(path)

    return tmp


@pytest.fixture(scope='session')
def tree():
    data = load_iris()
    X, y = data.data, data.target
    return DecisionTreeClassifier(random_state=0).fit(X=X, y=y)


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
    MLPClassifier,
    MLPRegressor,
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
    'MLPClassifier',
    'MLPRegressor',
])
def test_valid_base_estimator_since_0_14(Class: Callable):
    """Test initialization with valid base estimator."""
    if not Class:
        return
    est = Estimator(Class().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2]))
    assert isinstance(est.estimator, Class)


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18),
    reason='requires scikit-learn >= v0.18'
)
def test_list_of_regressors():
    """Test and compare list of regressors."""
    regressors = [
        MLPRegressor
    ]
    regressors = sorted([r.__class__.__name__ for r in regressors])
    candidates = sorted([r.__class__.__name__ for r in Estimator.regressors()])
    assert regressors == candidates


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18),
    reason='requires scikit-learn >= v0.18'
)
def test_list_of_classifiers():
    """Test and compare list of classifiers."""
    classifiers = [
        AdaBoostClassifier,
        BernoulliNB,
        DecisionTreeClassifier,
        ExtraTreesClassifier,
        GaussianNB,
        KNeighborsClassifier,
        LinearSVC,
        NuSVC,
        RandomForestClassifier,
        SVC,
        MLPClassifier,
    ]
    classifiers = sorted([c.__class__.__name__ for c in classifiers])
    candidates = sorted([c.__class__.__name__ for c in Estimator.classifiers()])
    assert classifiers == candidates


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
    MLPClassifier,
    MLPRegressor,
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
    'MLPClassifier',
    'MLPRegressor',
])
def test_unfitted_est(Class: Callable):
    """Test unfitted estimators."""
    if not Class:
        return
    with pytest.raises(NotFittedEstimatorError):
        Estimator(Class())


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
    pipeline.fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    est = Estimator(pipeline)
    assert isinstance(est.estimator, SVC)


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 15),
    reason='requires scikit-learn >= v0.15'
)
def test_unfitted_est_in_pipeline():
    """Test the extraction of an estimator from a pipeline."""
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('SVM', SVC())])
    with pytest.raises(NotFittedEstimatorError):
        Estimator(pipeline)


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
    MLPClassifier,
    MLPRegressor,
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
    'MLPClassifier',
    'MLPRegressor',
])
@pytest.mark.parametrize('params', [
    # (InvalidMethodError, dict(method='i_n_v_a_l_i_d')),
    (InvalidLanguageError, dict(language='i_n_v_a_l_i_d')),
    (InvalidTemplateError, dict(template='i_n_v_a_l_i_d')),
], ids=[
    # 'InvalidMethodError',
    'InvalidLanguageError',
    'InvalidTemplateError',
])
def test_invalid_params_on_port_method(Class: Callable, params: Tuple):
    """Test initialization with valid base estimator."""
    if not Class:
        return
    est = Class().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    with pytest.raises(params[0]):
        Estimator(est).port(**params[1])


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
    MLPClassifier,
    MLPRegressor,
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
    'MLPClassifier',
    'MLPRegressor',
])
@pytest.mark.parametrize('params', [
    # (InvalidMethodError, dict(method='i_n_v_a_l_i_d')),
    (InvalidLanguageError, dict(language='i_n_v_a_l_i_d')),
    (InvalidTemplateError, dict(template='i_n_v_a_l_i_d')),
], ids=[
    # 'InvalidMethodError',
    'InvalidLanguageError',
    'InvalidTemplateError',
])
def test_invalid_params_on_dump_method(Class: Callable, params: Tuple):
    """Test initialization with valid base estimator."""
    if not Class:
        return
    est = Class().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    with pytest.raises(params[0]):
        Estimator(est).dump(**params[1])


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
def test_extraction_from_optimizer(Class: Callable):
    """Test the extraction from an optimizer."""
    params = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 10, 100],
        'gamma': [0.001, 0.0001]
    }
    search = Class(SVC(), params, cv=2)

    # Test unfitted optimizer:
    with pytest.raises(ValueError):
        est = Estimator(search)
        assert isinstance(est.estimator, SVC)

    # Test fitted optimizer:
    search.fit(X=[[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
               y=[1, 2, 3, 1, 2, 3])
    est = Estimator(search)
    assert isinstance(est.estimator, SVC)


@pytest.mark.parametrize('x', [
    [5.1, 3.5, 1.4, 0.2],
    np.array([5.1, 3.5, 1.4, 0.2]),
    [[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]],
    np.array([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]]),
    [np.array([5.1, 3.5, 1.4, 0.2]), np.array([5.1, 3.5, 1.4, 0.2])],
], ids=[
    '{}_{}'.format(
        type([5.1, 3.5, 1.4, 0.2]).__qualname__,
        type([5.1, 3.5, 1.4, 0.2][0]).__qualname__
    ),
    '{}_{}'.format(
        type(np.array([5.1, 3.5, 1.4, 0.2])).__qualname__,
        type(np.array([5.1, 3.5, 1.4, 0.2])[0]).__qualname__
    ),
    '{}_{}'.format(
        type([
            [5.1, 3.5, 1.4, 0.2],
            [5.1, 3.5, 1.4, 0.2]
        ]).__qualname__,
        type([
                 [5.1, 3.5, 1.4, 0.2],
                 [5.1, 3.5, 1.4, 0.2]
             ][0]).__qualname__
    ),
    '{}_{}'.format(
        type(np.array([
            [5.1, 3.5, 1.4, 0.2],
            [5.1, 3.5, 1.4, 0.2]
        ])).__qualname__,
        type(np.array([
            [5.1, 3.5, 1.4, 0.2],
            [5.1, 3.5, 1.4, 0.2]
        ])[0]).__qualname__
    ),
    '{}_{}'.format(
        type([
            np.array([5.1, 3.5, 1.4, 0.2]),
            np.array([5.1, 3.5, 1.4, 0.2])
        ]).__qualname__,
        type([
                 np.array([5.1, 3.5, 1.4, 0.2]),
                 np.array([5.1, 3.5, 1.4, 0.2])
             ][0]).__qualname__
    ),
])
@pytest.mark.parametrize('template', [
    'attached',
    'combined',
    'exported',
])
@pytest.mark.parametrize('language', [
    'c',
    'go',
    'java',
    'js',
    'php',
    'ruby'
])
def test_make_inputs_outputs(tmp: Path, x, tree, template: str, language: str):
    est = Estimator(tree)

    if not est.support(
            language=language,
            template=template,
            method='predict'
    ):
        return

    def fs_mkdir(base_dir: Path, test_name: str,
                 estimator_name: str,
                 language_name: str, template_name: str) -> Path:
        """Helper function to create a directory for tests."""
        base_dir = base_dir \
                   / ('test__' + test_name) \
                   / ('estimator__' + estimator_name) \
                   / ('language__' + language_name) \
                   / ('template__' + template_name)
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    tmp = fs_mkdir(
        base_dir=tmp, test_name='test_make_inputs_outputs',
        estimator_name='DecisionTreeClassifier',
        language_name=language, template_name=template
    )
    out = est.make(x, language=language, template=template,
                   directory=tmp, n_jobs=False)

    assert isinstance(out, tuple)
    assert len(out) == 2
    y_pred, y_proba = out
    if isinstance(x[0], (int, float)):
        assert isinstance(y_pred, np.int64)
        assert y_pred == 0
        assert isinstance(y_proba, np.ndarray)
        assert list(y_proba) == [1, 0, 0]
    if isinstance(x[0], (list, np.ndarray)):
        assert isinstance(y_pred, np.ndarray)
        assert y_pred[0] == 0
        assert isinstance(y_proba, np.ndarray)
        assert list(y_proba[0]) == [1, 0, 0]


@pytest.mark.parametrize('template', [
    'attached',
    'combined',
    'exported',
])
@pytest.mark.parametrize('language', [
    'c',
    'go',
    'java',
    'js',
    'php',
    'ruby',
])
@pytest.mark.parametrize('dataset', [
    ('iris', load_iris()),
    ('breast_cancer', load_breast_cancer()),
    ('digits', load_digits()),
], ids=[
    'iris',
    'breast_cancer',
    'digits',
])
@pytest.mark.parametrize('Class', [
    ('DecisionTreeClassifier', DecisionTreeClassifier, dict(random_state=0)),
], ids=[
    'DecisionTreeClassifier',
])
def test_and_compare_accuracies(
        tmp: Path,
        Class: Optional[Tuple[str, Callable, Dict]],
        dataset: Tuple, template: str, language: str):
    """Test a wide range of variations."""
    if not Class or not dataset[1]:
        return

    def fs_mkdir(base_dir: Path, test_name: str,
                 estimator_name: str, dataset_name: str,
                 language_name: str, template_name: str) -> Path:
        """Helper function to create a directory for tests."""
        base_dir = base_dir \
                   / ('test__' + test_name) \
                   / ('estimator__' + estimator_name) \
                   / ('dataset__' + dataset_name) \
                   / ('language__' + language_name) \
                   / ('template__' + template_name)
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def ds_generate_x(x: np.ndarray, n_samples: int) -> np.ndarray:
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'Two dimensional numpy array is required.'
            raise AssertionError(msg)
        return np.random.uniform(
            low=np.amin(x, axis=0),
            high=np.amax(x, axis=0),
            size=(n_samples, len(x[0]))
        )

    def ds_uniform_x(x: np.ndarray, n_samples: int) -> np.ndarray:
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'Two dimensional numpy array is required.'
            raise AssertionError(msg)
        n_samples = min(n_samples, len(x))
        return x[(np.random.uniform(0, 1, n_samples)
                  * (len(x) - 1)).astype(int)]

    orig_est = Class[1](**Class[2])
    x, y = dataset[1].data, dataset[1].target
    orig_est.fit(X=x, y=y)
    try:
        est = Estimator(orig_est)
        if est.support(
                language=language,
                template=template,
                method='predict'
        ):
            tmp = fs_mkdir(
                base_dir=tmp, test_name='test_and_compare_accuracies',
                estimator_name=Class[0], dataset_name=dataset[0],
                language_name=language, template_name=template
            )
            est.dump(
                language=language,
                template=template,
                directory=tmp
            )

            # Generate test data:
            synt_x = ds_generate_x(x, 100)
            unif_x = ds_uniform_x(x, 100)

    except:
        pytest.fail('Unexpected exception ...')
