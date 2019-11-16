# -*- coding: utf-8 -*-

import random as rd
import shutil
import urllib.request
from os import environ
from pathlib import Path
from sys import version_info as PYTHON_VERSION
from typing import Tuple, List
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

# sklearn-porter
from sklearn_porter.language.Java import Java
from sklearn_porter.Estimator import Estimator
from sklearn_porter import exceptions as exception

# List of named tuples:
Dataset = namedtuple('Dataset', 'name data target')
Candidate = namedtuple('candidate', 'name clazz create')

# Force deterministic number generation:
np.random.seed(0)
rd.seed(0)

# Check python version:
if PYTHON_VERSION[:2] < (3, 5):
    pytest.skip('tests requires python >= 3.5', allow_module_level=True)

# Parse and prepare scikit-learn version:
SKLEARN_VERSION = tuple(map(int, str(sklearn.__version__).split('.')))

environ['SKLEARN_PORTER_PYTEST'] = 'True'


def get_candidates() -> List[Candidate]:
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

    _regressors = []
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPRegressor
    except ImportError:
        pass
    else:
        _regressors.append(MLPRegressor)

    _estimators = _classifiers + _regressors

    for e in _estimators:
        yield (Candidate(e.__name__, e, e))


def get_searchers() -> List[Candidate]:
    _searchers = []
    try:
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        pass
    else:
        _searchers.append(
            Candidate(GridSearchCV.__name__, GridSearchCV, GridSearchCV)
        )
    try:
        from sklearn.model_selection import RandomizedSearchCV
    except ImportError:
        pass
    else:
        _searchers.append(
            Candidate(
                RandomizedSearchCV.__name__, RandomizedSearchCV,
                RandomizedSearchCV
            )
        )
    return _searchers


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


candidates = list(get_candidates())
searchers = list(get_searchers())
datasets = list(get_datasets())


@pytest.fixture(scope='session')
def tmp_root_dir(worker_id) -> Path:
    """Fixture to get the path to the temporary directory."""

    # Delete previous generated temporary directory:
    tmp_dir = Path(__file__).parent / 'tmp'
    if worker_id is 'master':
        if tmp_dir.exists():
            shutil.rmtree(str(tmp_dir.resolve()), ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src_dir = (Path(__file__).parent / '..').resolve()
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


@pytest.mark.parametrize('candidate', candidates, ids=lambda x: x.name)
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


@pytest.mark.parametrize('candidate', candidates, ids=lambda x: x.name)
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


@pytest.mark.parametrize('candidate', candidates, ids=lambda x: x.name)
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
def test_invalid_params_on_dump_method(candidate: Candidate, params: Tuple):
    """Test initialization with valid base estimator."""
    est = candidate.create().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    with pytest.raises(params[0]):
        Estimator(est).dump(**params[1])


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 19), reason='requires scikit-learn >= v0.19'
)
@pytest.mark.parametrize('candidate', searchers, ids=lambda x: x.name)
def test_extraction_from_optimizer(candidate: Candidate):
    """Test the extraction from an optimizer."""
    params = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 10, 100],
        'gamma': [0.001, 0.0001],
    }
    search = candidate.create(SVC(), params, cv=2)

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
    if not Estimator.can(fitted_tree, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

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

    est = Estimator(fitted_tree)
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
@pytest.mark.parametrize('candidate', candidates, ids=lambda x: x.name)
def test_and_compare_accuracies(
    tmp_root_dir: Path,
    candidate: Candidate,
    dataset: Dataset,
    template: str,
    language: str,
):
    if not Estimator.can(candidate.clazz, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    def fs_mkdir(
        base_dir: Path,
        test_name: str,
        estimator_name: str,
        dataset_name: str,
        language_name: str,
        template_name: str,
    ) -> Path:
        """
        Helper function to create a separate
        directory for generated source files.
        """
        base_dir = (
            base_dir / ('test__' + test_name) /
            ('estimator__' + estimator_name) / ('dataset__' + dataset_name) /
            ('language__' + language_name) / ('template__' + template_name)
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def ds_generate_x(x: np.ndarray, n_samples: int) -> np.ndarray:
        """Helper function to create uniform test samples."""
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'Two dimensional numpy array is required.'
            raise AssertionError(msg)
        return np.random.uniform(
            low=np.amin(x, axis=0),
            high=np.amax(x, axis=0),
            size=(n_samples, len(x[0])),
        )

    def ds_uniform_x(x: np.ndarray, n_samples: int) -> np.ndarray:
        """Helper function to pick random test samples."""
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'Two dimensional numpy array is required.'
            raise AssertionError(msg)
        n_samples = min(n_samples, len(x))
        return x[(np.random.uniform(0, 1, n_samples) *
                  (len(x) - 1)).astype(int)]

    # Estimator:
    orig_est = candidate.create()
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    n_uni = int(environ.get('SKLEARN_PORTER_PYTEST_N_UNI_REGRESSION_TESTS', 15))
    n_gen = int(environ.get('SKLEARN_PORTER_PYTEST_N_GEN_REGRESSION_TESTS', 15))
    test_x = np.vstack((ds_uniform_x(x, n_uni), ds_generate_x(x, n_gen)))

    # Directory:
    tmp_dir = fs_mkdir(
        base_dir=tmp_root_dir,
        test_name='test_and_compare_accuracies',
        estimator_name=candidate.name,
        dataset_name=dataset.name,
        language_name=language,
        template_name=template,
    )

    try:
        score = est.integrity_score(
            test_x, language=language, template=template, directory=tmp_dir
        )
        assert score == 1.
    except exception.CodeTooLarge:
        warn_msg = 'Code too large for the combination: ' \
                   'estimator: {}, language: {}, template: {}, dataset: {}' \
                   ''.format(candidate.name, language, template, dataset.name)
        warnings.warn(warn_msg)
    except:
        pytest.fail('Unexpected exception ...')
