# -*- coding: utf-8 -*-

import random as rd
import shutil
import urllib.request
from os import environ
from pathlib import Path
from sys import version_info as PYTHON_VERSION
from typing import Callable, Dict, Optional, Tuple
from collections import namedtuple
import warnings

import numpy as np
import pytest

# scikit-learn
import sklearn
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble.forest import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm.classes import SVC, LinearSVC, NuSVC
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn_porter.Estimator import Estimator
from sklearn_porter.exceptions import (
    InvalidLanguageError, InvalidTemplateError, NotFittedEstimatorError,
    CodeTooLarge
)

# sklearn-porter
from sklearn_porter.language.Java import Java

classifiers = [
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
    classifiers.append(MLPClassifier)

regressors = []
try:
    from sklearn.neural_network.multilayer_perceptron import MLPRegressor
except ImportError:
    pass
else:
    regressors.append(MLPRegressor)

estimators = classifiers + regressors

searchers = []
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    pass
else:
    searchers.append(GridSearchCV)
try:
    from sklearn.model_selection import RandomizedSearchCV
except ImportError:
    pass
else:
    searchers.append(RandomizedSearchCV)

Dataset = namedtuple('Dataset', 'name data target')
datasets = [
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
    datasets.append(
        Dataset(
            'breast_cancer',
            load_breast_cancer().data,
            load_breast_cancer().target
        )
    )

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
def tmp_root_dir(worker_id) -> Path:
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
def fitted_tree() -> DecisionTreeClassifier:
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X=x, y=y)
    return tree


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18), reason='requires scikit-learn >= v0.18'
)
def test_list_of_regressors():
    """Test and compare list of regressors."""
    regressors = [MLPRegressor]
    regressors = sorted([r.__class__.__name__ for r in regressors])
    candidates = sorted([r.__class__.__name__ for r in Estimator.regressors()])
    assert regressors == candidates


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18), reason='requires scikit-learn >= v0.18'
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


@pytest.mark.parametrize('Class', estimators, ids=lambda x: x.__qualname__)
def test_unfitted_estimator(Class: Callable):
    """Test unfitted estimators."""
    with pytest.raises(NotFittedEstimatorError):
        Estimator(Class())


@pytest.mark.parametrize('obj', [None, object, 0, 'string'])
def test_invalid_estimator(obj):
    """Test initialization with invalid base estimator."""
    with pytest.raises(ValueError):
        Estimator(obj)


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 15), reason='requires scikit-learn >= v0.15'
)
def test_estimator_extraction_from_pipeline():
    """Test the extraction of an estimator from a pipeline."""
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('SVM', SVC())])
    pipeline.fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    est = Estimator(pipeline)
    assert isinstance(est.estimator, SVC)


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 15), reason='requires scikit-learn >= v0.15'
)
def test_unfitted_estimator_in_pipeline():
    """Test the extraction of an estimator from a pipeline."""
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('SVM', SVC())])
    with pytest.raises(NotFittedEstimatorError):
        Estimator(pipeline)


@pytest.mark.parametrize('Class', estimators, ids=lambda x: x.__qualname__)
@pytest.mark.parametrize(
    'params',
    [
        (InvalidLanguageError, dict(language='i_n_v_a_l_i_d')),
        (InvalidTemplateError, dict(template='i_n_v_a_l_i_d')),
    ],
    ids=[
        'InvalidLanguageError',
        'InvalidTemplateError',
    ],
)
def test_invalid_params_on_port_method(Class: Callable, params: Tuple):
    """Test initialization with valid base estimator."""
    est = Class().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    with pytest.raises(params[0]):
        Estimator(est).port(**params[1])


@pytest.mark.parametrize('Class', estimators, ids=lambda x: x.__qualname__)
@pytest.mark.parametrize(
    'params',
    [
        (InvalidLanguageError, dict(language='i_n_v_a_l_i_d')),
        (InvalidTemplateError, dict(template='i_n_v_a_l_i_d')),
    ],
    ids=[
        'InvalidLanguageError',
        'InvalidTemplateError',
    ],
)
def test_invalid_params_on_dump_method(Class: Callable, params: Tuple):
    """Test initialization with valid base estimator."""
    est = Class().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    with pytest.raises(params[0]):
        Estimator(est).dump(**params[1])


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 19), reason='requires scikit-learn >= v0.19'
)
@pytest.mark.parametrize('Class', searchers, ids=lambda x: x.__qualname__)
def test_extraction_from_optimizer(Class: Callable):
    """Test the extraction from an optimizer."""
    params = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 10, 100],
        'gamma': [0.001, 0.0001],
    }
    search = Class(SVC(), params, cv=2)

    # Test unfitted optimizer:
    with pytest.raises(ValueError):
        est = Estimator(search)
        assert isinstance(est.estimator, SVC)

    # Test fitted optimizer:
    search.fit(
        X=[[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
        y=[1, 2, 3, 1, 2, 3]
    )
    est = Estimator(search)
    assert isinstance(est.estimator, SVC)


@pytest.mark.parametrize(
    'x',
    [
        [5.1, 3.5, 1.4, 0.2],
        np.array([5.1, 3.5, 1.4, 0.2]),
        [[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]],
        np.array([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]]),
        [np.array([5.1, 3.5, 1.4, 0.2]),
         np.array([5.1, 3.5, 1.4, 0.2])],
    ],
    ids=[  # fmt: off
        '{}_{}'.format(
            type([5.1, 3.5, 1.4, 0.2]).__qualname__,
            type([5.1, 3.5, 1.4, 0.2][0]).__qualname__,
        ),
        '{}_{}'.format(
            type(np.array([5.1, 3.5, 1.4, 0.2])).__qualname__,
            type(np.array([5.1, 3.5, 1.4, 0.2])[0]).__qualname__,
        ),
        '{}_{}'.format(
            type([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]]).__qualname__,
            type([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]][0]).__qualname__,
        ),
        '{}_{}'.format(
            type(np.array([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]])).__qualname__,
            type(np.array([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]])[0]).__qualname__,
        ),
        '{}_{}'.format(
            type([np.array([5.1, 3.5, 1.4, 0.2]), np.array([5.1, 3.5, 1.4, 0.2])]).__qualname__,
            type([np.array([5.1, 3.5, 1.4, 0.2]), np.array([5.1, 3.5, 1.4, 0.2])][0]).__qualname__,
        ),
    ],  # fmt: on
)
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
def test_make_inputs_outputs(
    tmp_root_dir: Path, x, fitted_tree: DecisionTreeClassifier, template: str,
    language: str
):
    est = Estimator(fitted_tree)

    if not est.support(language=language, template=template, method='predict'):
        return

    def fs_mkdir(
        base_dir: Path,
        test_name: str,
        estimator_name: str,
        language_name: str,
        template_name: str,
    ) -> Path:
        """Helper function to create a directory for tests."""
        base_dir = (
            base_dir / ('test__' + test_name) /
            ('estimator__' + estimator_name) / ('language__' + language_name) /
            ('template__' + template_name)
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    tmp_dir = fs_mkdir(
        base_dir=tmp_root_dir,
        test_name='test_make_inputs_outputs',
        estimator_name='DecisionTreeClassifier',
        language_name=language,
        template_name=template,
    )
    out = est.make(
        x,
        language=language,
        template=template,
        directory=tmp_dir,
        n_jobs=False
    )

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


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', datasets, ids=lambda x: x.name)
@pytest.mark.parametrize(
    'Class',
    [  # fmt: off
        ('DecisionTreeClassifier', DecisionTreeClassifier, dict(random_state=0)),
        ('BernoulliNB', BernoulliNB, dict()),
        ('RandomForestClassifier', RandomForestClassifier, dict()),
        ('ExtraTreesClassifier', ExtraTreesClassifier, dict()),
        ('LinearSVC', LinearSVC, dict()),
        ('SVC', SVC, dict()),
        ('NuSVC', NuSVC, dict()),
        ('KNeighborsClassifier', KNeighborsClassifier, dict()),
        ('GaussianNB', GaussianNB, dict()),
    ],  # fmt: on
    ids=[
        'DecisionTreeClassifier',
        'BernoulliNB',
        'RandomForestClassifier',
        'ExtraTreesClassifier',
        'LinearSVC',
        'SVC',
        'NuSVC',
        'KNeighborsClassifier',
        'GaussianNB'
    ]
)
def test_and_compare_accuracies(
    tmp_root_dir: Path,
    Class: Optional[Tuple[str, Callable, Dict]],
    dataset: Dataset,
    template: str,
    language: str,
):
    """Test a wide range of variations."""
    def fs_mkdir(
        base_dir: Path,
        test_name: str,
        estimator_name: str,
        dataset_name: str,
        language_name: str,
        template_name: str,
    ) -> Path:
        """Helper function to create a directory for tests."""
        base_dir = (
            base_dir / ('test__' + test_name) /
            ('estimator__' + estimator_name) / ('dataset__' + dataset_name) /
            ('language__' + language_name) / ('template__' + template_name)
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def ds_generate_x(x: np.ndarray, n_samples: int) -> np.ndarray:
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'Two dimensional numpy array is required.'
            raise AssertionError(msg)
        return np.random.uniform(
            low=np.amin(x, axis=0),
            high=np.amax(x, axis=0),
            size=(n_samples, len(x[0])),
        )

    def ds_uniform_x(x: np.ndarray, n_samples: int) -> np.ndarray:
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'Two dimensional numpy array is required.'
            raise AssertionError(msg)
        n_samples = min(n_samples, len(x))
        return x[(np.random.uniform(0, 1, n_samples) *
                  (len(x) - 1)).astype(int)]

    # Dataset:
    x, y = dataset.data, dataset.target

    # Estimator:
    orig_est = Class[1](**Class[2])
    orig_est.fit(X=x, y=y)
    try:
        # Porter:
        est = Estimator(orig_est)
        if est.support(language=language, template=template, method='predict'):
            tmp_dir = fs_mkdir(
                base_dir=tmp_root_dir,
                test_name='test_and_compare_accuracies',
                estimator_name=Class[0],
                dataset_name=dataset.name,
                language_name=language,
                template_name=template,
            )
            test_x = np.vstack((ds_uniform_x(x, 10), ds_generate_x(x, 10)))
            score = est.integrity_score(
                test_x, language=language, template=template, directory=tmp_dir
            )
            assert score == 1.
    except CodeTooLarge:
        warn_msg = 'Code too large for the combination: ' \
                   'estimator: {}, language: {}, template: {}, dataset: {}' \
                   ''.format(Class[0], language, template, dataset.name)
        warnings.warn(warn_msg)
    except:
        pytest.fail('Unexpected exception ...')
