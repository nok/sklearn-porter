# -*- coding: utf-8 -*-

import random as rd
import shutil
import urllib.request
import warnings
from collections import namedtuple
from os import environ
from pathlib import Path
from sys import version_info as PYTHON_VERSION
from typing import List, Tuple, Union

import numpy as np
import pytest

# scikit-learn
import sklearn
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble.forest import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.cli.__main__ import parse_args
from sklearn_porter.Estimator import Estimator, can
from sklearn_porter.language.Java import Java
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Force deterministic number generation:
np.random.seed(0)
rd.seed(0)

# Check python version:
if PYTHON_VERSION[:2] < (3, 5):
    pytest.skip('tests requires python >= 3.5', allow_module_level=True)

# List of named tuples:
Dataset = namedtuple('Dataset', 'name data target')
Candidate = namedtuple('Candidate', 'name clazz create')

# Parse and prepare scikit-learn version:
SKLEARN_VERSION = tuple(map(int, str(sklearn.__version__).split('.')))

environ['SKLEARN_PORTER_PYTEST'] = 'True'

FILE_DIR = Path(__file__).parent
SERIALIZED_MODEL = FILE_DIR / '..' / 'examples' / 'recipes' / 'dump_estimator_to_pickle_file' / 'estimator.pkl'


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


test_classifiers = list(get_classifiers())
test_regressors = list(get_regressors())
test_candidates = test_classifiers + test_regressors
test_datasets = list(get_datasets())


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


@pytest.mark.parametrize('candidate', test_candidates, ids=lambda x: x.name)
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


@pytest.mark.parametrize('candidate', test_candidates, ids=lambda x: x.name)
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


@pytest.mark.parametrize('candidate', test_candidates, ids=lambda x: x.name)
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
            ('test', 'test_make_the_inputs_outputs'),
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
@pytest.mark.parametrize('dataset', test_datasets, ids=lambda x: x.name)
@pytest.mark.parametrize('candidate', test_classifiers, ids=lambda x: x.name)
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
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir, [
            ('test', 'test_regressions_of_classifiers'),
            ('estimator', candidate.name),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ]
    )

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
        msg = 'Code too large for the combination: ' \
              'estimator: {}, language: {}, template: {}, dataset: {}' \
              ''.format(candidate.name, language, template, dataset.name)
        warnings.warn(msg)
    except:
        pytest.fail('Unexpected exception ...')


@pytest.mark.parametrize(
    'args',
    [['show'], ['port', str(SERIALIZED_MODEL), '--skip-warnings']],
    ids=['show', 'port']
)
def test_cli_subcommand(args: List):
    parsed = parse_args(args)
    cmd = args[0]
    assert (parsed.cmd == cmd)


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18), reason='requires scikit-learn >= v0.18'
)
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('activation', ['relu', 'identity', 'tanh', 'logistic'])
@pytest.mark.parametrize(
    'hidden_layer_sizes', [15, [15, 5], [15, 10, 5]],
    ids=['15', '15_5', '15_10_5']
)
@pytest.mark.parametrize('dataset', test_datasets, ids=lambda x: x.name)
def test_mlp_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    activation: str,
    hidden_layer_sizes: Union[int, List[int]],
):
    """Test and compare different MLP classifiers."""
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPClassifier
    except ImportError:
        pass
    else:
        orig_est = MLPClassifier(
            activation=activation,
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=.1,
            random_state=1,
            max_iter=10,
        )

        if not can(orig_est, language, template, 'predict'):
            pytest.skip(
                'Skip unsupported estimator/'
                'language/template combination'
            )

        # Estimator:
        x, y = dataset.data, dataset.target
        orig_est.fit(X=x, y=y)
        est = Estimator(orig_est)

        # Samples:
        test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

        # Directory:
        hls = hidden_layer_sizes  # alias
        hls = '_'.join(
            [str(hls)] if isinstance(hls, int) else list(map(str, hls))
        )
        tmp_dir = fs_mkdir(
            tmp_root_dir, [
                ('test', 'test_mlp_classifier'), ('dataset', dataset.name),
                ('language', language), ('template', template),
                ('activation', activation), ('hidden_layer_sizes', hls)
            ]
        )

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
            msg = 'Code too large for the combination: ' \
                  'language: {}, template: {}, dataset: {}' \
                  ''.format(language, template, dataset.name)
            warnings.warn(msg)
        except:
            pytest.fail('Unexpected exception ...')
