# -*- coding: utf-8 -*-
import os
import random as rd
import shutil
import urllib.request
import warnings
from os import environ
from pathlib import Path
from sys import version_info
from typing import Callable, List, Tuple

import numpy as np
import pytest

# scikit-learn
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.ensemble.forest import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.cli.__main__ import parse_args
from sklearn_porter.Estimator import Estimator, can, show
from sklearn_porter.language.Java import Java

from tests.commons import (
    CANDIDATES, CLASSIFIERS, DATASETS, SKLEARN_VERSION, TESTS_DIR, Candidate,
    Dataset
)
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Force deterministic number generation:
np.random.seed(0)
rd.seed(0)

# Check python version:
if version_info[:2] < (3, 5):
    pytest.skip('tests requires python >= 3.5', allow_module_level=True)

environ['SKLEARN_PORTER_PYTEST'] = 'True'

SERIALIZED_MODEL = TESTS_DIR / 'resources' / 'estimator_0_19.pkl'


@pytest.fixture(scope='session')
def tmp_root_dir(worker_id) -> Path:
    """Fixture to get the path to the temporary directory."""

    # Delete previous generated temporary directory:
    tmp_dir = TESTS_DIR / 'tmp'
    if worker_id is 'master':
        if tmp_dir.exists():
            shutil.rmtree(str(tmp_dir.resolve()), ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src_dir = (TESTS_DIR / '..').resolve()
    gson_fname = 'gson.jar'
    src_gson_path = src_dir / gson_fname
    tmp_gson_path = tmp_dir / gson_fname

    # Download missing dependencies:
    if not src_gson_path.exists():
        url = Java.GSON_DOWNLOAD_URI
        urllib.request.urlretrieve(url, str(src_gson_path))

    shutil.copy(str(src_gson_path), str(tmp_gson_path))
    environ['SKLEARN_PORTER_PYTEST_GSON_PATH'] = str(tmp_gson_path)

    return tmp_dir


@pytest.fixture(scope='session')
def fitted_tree() -> DecisionTreeClassifier:
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X=x, y=y)
    return tree


def test_unsupported_estimator():
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    clf = SGDClassifier()
    clf.fit(x, y)
    with pytest.raises(exception.NotSupportedYetError):
        Estimator(clf)


def test_unsupported_objects():
    x = [[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]
    binarizer = preprocessing.Binarizer().fit(x)
    with pytest.raises(ValueError):
        Estimator(binarizer)


def test_valid_but_unsupported_language():
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    clf = AdaBoostClassifier()
    clf.fit(X=x, y=y)
    with pytest.raises(exception.NotSupportedYetError):
        Estimator(clf, language='ruby')


def test_valid_but_unsupported_template(fitted_tree):
    with pytest.raises(exception.NotSupportedYetError):
        Estimator(fitted_tree, language='c', template='exported')


def test_repr_output(fitted_tree):
    est = Estimator(fitted_tree)
    assert 'DecisionTreeClassifier' in str(est)


def test_show_output():
    table = show()
    assert 'DecisionTreeClassifier' in table
    assert 'Java' in table
    assert 'JavaScript' in table

    table = show(language='c')
    assert 'DecisionTreeClassifier' in table
    assert 'Java' not in table
    assert 'JavaScript' not in table


def test_optional_kwargs(fitted_tree):
    est = Estimator(
        fitted_tree,
        class_name='Estimator',
        converter=lambda x: '{:.3f}'.format(x)
    )
    assert est.class_name == 'Estimator'
    assert est.converter(0.11111) == '0.111'


def test_common_properties(fitted_tree):
    est = Estimator(fitted_tree)
    assert isinstance(est.template, str)
    assert isinstance(est.language, str)
    assert isinstance(est.class_name, str)
    assert isinstance(est.converter, Callable)
    assert isinstance(est.estimator, BaseEstimator)


def test_and_compare_multiprocessed_results(fitted_tree):
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    est = Estimator(fitted_tree)
    res_a = est.make(x[:3], n_jobs=False)
    res_b = est.make(x[:3], n_jobs=True)
    assert res_a[0][0] == res_b[0][0]
    assert res_a[0][1] == res_b[0][1]
    assert res_a[0][2] == res_b[0][2]


def test_file_handling(fitted_tree, tmp_root_dir):
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    est = Estimator(fitted_tree, language='java')

    # Delete generated files:
    tmp_dir_delete = fs_mkdir(
        tmp_root_dir, [
            ('test', 'file_handling_delete'),
        ]
    )
    est.make(
        x[:3],
        n_jobs=False,
        directory=tmp_dir_delete,
        delete_created_files=True
    )
    assert len(os.listdir(str(tmp_dir_delete))) == 0

    # Keep generated files:
    tmp_dir_keep = fs_mkdir(tmp_root_dir, [
        ('test', 'file_handling_keep'),
    ])
    est.make(
        x[:3], n_jobs=False, directory=tmp_dir_keep, delete_created_files=False
    )
    assert len(os.listdir(str(tmp_dir_keep))) == 2


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18), reason='requires scikit-learn >= v0.18'
)
def test_list_of_regressors():
    """Test and compare list of regressors."""
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPRegressor
    except ImportError:
        pass
    else:
        regressors = [MLPRegressor]
        regressors = sorted([r.__class__.__name__ for r in regressors])
        result = sorted([r.__class__.__name__ for r in Estimator.regressors()])
        assert regressors == result


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18), reason='requires scikit-learn >= v0.18'
)
def test_list_of_classifiers():
    """Test and compare list of classifiers."""
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPClassifier
    except ImportError:
        pass
    else:
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
        result = sorted([c.__class__.__name__ for c in Estimator.classifiers()])
        assert classifiers == result


@pytest.mark.parametrize('candidate', CANDIDATES, ids=lambda x: x.name)
def test_unfitted_estimator(candidate: Candidate):
    """Test unfitted estimators."""
    with pytest.raises(exception.NotFittedEstimatorError):
        Estimator(candidate.create())


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
    with pytest.raises(exception.NotFittedEstimatorError):
        Estimator(pipeline)


@pytest.mark.parametrize('candidate', CANDIDATES, ids=lambda x: x.name)
@pytest.mark.parametrize(
    'params',
    [
        (exception.InvalidLanguageError, dict(language='i_n_v_a_l_i_d')),
        (exception.InvalidTemplateError, dict(template='i_n_v_a_l_i_d')),
    ],
    ids=[
        'InvalidLanguageError',
        'InvalidTemplateError',
    ],
)
def test_invalid_params_on_port_method(candidate: Candidate, params: Tuple):
    """Test initialization with valid base estimator."""
    est = candidate.create().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    with pytest.raises(params[0]):
        Estimator(est).port(**params[1])


@pytest.mark.parametrize('candidate', CANDIDATES, ids=lambda x: x.name)
@pytest.mark.parametrize(
    'params',
    [
        (exception.InvalidLanguageError, dict(language='i_n_v_a_l_i_d')),
        (exception.InvalidTemplateError, dict(template='i_n_v_a_l_i_d')),
    ],
    ids=[
        'InvalidLanguageError',
        'InvalidTemplateError',
    ],
)
def test_invalid_params_on_save_method(candidate: Candidate, params: Tuple):
    """Test initialization with valid base estimator."""
    est = candidate.create().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    with pytest.raises(params[0]):
        Estimator(est).save(**params[1])


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 19), reason='requires scikit-learn >= v0.19'
)
def test_extraction_from_grid_search_cv():
    """Test the extraction from an optimizer."""
    from sklearn.model_selection import GridSearchCV

    params = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 10, 100],
        'gamma': [0.001, 0.0001],
    }
    search = GridSearchCV(SVC(), params, cv=2)

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


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 19), reason='requires scikit-learn >= v0.19'
)
def test_extraction_from_randomized_search_cv():
    """Test the extraction from an optimizer."""
    from sklearn.model_selection import RandomizedSearchCV

    params = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 10, 100],
        'gamma': [0.001, 0.0001],
    }
    search = RandomizedSearchCV(SVC(), params, cv=2)

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
def test_make_the_inputs_outputs(
    tmp_root_dir: Path, x, fitted_tree: DecisionTreeClassifier, template: str,
    language: str
):
    if not can(fitted_tree, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir, [
            ('test', 'make_the_inputs_outputs'),
            ('estimator', 'DecisionTreeClassifier'),
            ('language', language),
            ('template', template),
        ]
    )

    # Estimator:
    est = Estimator(fitted_tree)
    out = est.make(
        x,
        language=language,
        template=template,
        directory=tmp_dir,
        n_jobs=False,
        delete_created_files=False
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
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('candidate', CLASSIFIERS, ids=lambda x: x.name)
def test_regressions_of_classifiers(
    tmp_root_dir: Path,
    candidate: Candidate,
    dataset: Dataset,
    template: str,
    language: str,
):
    if not can(candidate.clazz(), language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Estimator:
    orig_est = candidate.create()
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)

    warnings.filterwarnings('error')
    try:
        est = Estimator(orig_est)
    except exception.QualityWarning as w:
        pytest.skip(str(w))

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir, [
            ('test', 'regressions_of_classifiers'),
            ('estimator', candidate.name),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ]
    )

    warnings.filterwarnings('default')
    try:
        score = est.test(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False
        )
        assert score == 1.
    except exception.CodeTooLarge:
        msg = 'The code too large for the combination: ' \
              'estimator: {}, language: {}, template: {}, dataset: {}' \
              ''.format(candidate.name, language, template, dataset.name)
        warnings.warn(msg)
    except exception.TooManyConstants:
        msg = 'The code has too many constants and can not be compiled: ' \
              'estimator: {}, language: {}, template: {}, dataset: {}' \
              ''.format(candidate.name, language, template, dataset.name)
        warnings.warn(msg)
    except Exception as e:
        pytest.fail('Unexpected exception ... ' + str(e))


@pytest.mark.parametrize(
    'args', [
        ['show'], ['show', '-l', 'java'], ['show', '-l', 'js'],
        ['port', str(SERIALIZED_MODEL), '--skip-warnings'],
        ['save', str(SERIALIZED_MODEL), '--skip-warnings']
    ],
    ids=['show', 'show_java', 'show_js', 'port', 'save']
)
def test_cli_subcommand(args: List):
    parsed = parse_args(args)
    cmd = args[0]
    assert (parsed.cmd == cmd)
