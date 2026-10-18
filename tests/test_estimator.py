import json
import os
import warnings
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from subprocess import CompletedProcess, run

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

# scikit-learn
from sklearn import preprocessing
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    make_multilabel_classification,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Perceptron,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.cli import utils as cli_utils
from sklearn_porter.cli.__main__ import parse_args
from sklearn_porter.cli.command.port import main as port_main
from sklearn_porter.cli.command.save import main as save_main
from sklearn_porter.cli.command.show import main as show_main
from sklearn_porter.Estimator import Estimator, can, show
from sklearn_porter.estimator.AdaBoostClassifier import (
    AdaBoostClassifier as AdaBoostClassifierPorter,
)
from tests.commons import (
    CANDIDATES,
    CLASSIFIERS,
    DATASETS,
    REGRESSION_DATASETS,
    REGRESSORS,
    Candidate,
    Dataset,
)
from tests.utils import (
    dataset_generate_x,
    dataset_uniform_x,
    drop_neighbor_ties,
    fs_mkdir,
)

# The module itself (the package exports the class of the same name):
ESTIMATOR_MODULE = import_module('sklearn_porter.Estimator')


def test_unsupported_estimator():
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    clf = NearestCentroid()
    clf.fit(x, y)
    with pytest.raises(exception.NotSupportedYetError):
        Estimator(clf)


def test_unsupported_objects():
    x = [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
    binarizer = preprocessing.Binarizer().fit(x)
    with pytest.raises(ValueError):
        Estimator(binarizer)


def test_valid_but_unsupported_language(monkeypatch: pytest.MonkeyPatch):
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    clf = AdaBoostClassifier()
    clf.fit(X=x, y=y)
    # All estimators support all languages, so remove one temporarily:
    support = dict(AdaBoostClassifierPorter.SUPPORT)
    support.pop(enum.Language.RUBY)
    monkeypatch.setattr(AdaBoostClassifierPorter, 'SUPPORT', support)
    with pytest.raises(exception.NotSupportedYetError):
        Estimator(clf, language='ruby')


def test_valid_but_unsupported_template():
    dataset = load_iris()
    x, y = dataset.data, dataset.target
    clf = KNeighborsClassifier()
    clf.fit(X=x, y=y)
    with pytest.raises(exception.NotSupportedYetError):
        Estimator(clf, language='c', template='combined')


def test_repr_output(fitted_tree):
    est = Estimator(fitted_tree)
    assert 'DecisionTreeClassifier' in str(est)


def test_show_output():
    table = show()
    assert 'DecisionTreeClassifier' in table
    assert 'IsolationForest' in table
    assert 'Java' in table
    assert 'JavaScript' in table

    table = show(language='c')
    assert 'DecisionTreeClassifier' in table
    assert 'Java' not in table
    assert 'JavaScript' not in table


def test_optional_kwargs(fitted_tree):
    est = Estimator(
        fitted_tree, class_name='Estimator', converter=lambda x: f'{x:.3f}'
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


@pytest.mark.regression
def test_and_compare_multiprocessed_results(fitted_tree):
    x = load_iris().data
    est = Estimator(fitted_tree)
    res_a = est.make(x[:3], n_jobs=False)
    res_b = est.make(x[:3], n_jobs=True)
    assert res_a[0][0] == res_b[0][0]
    assert res_a[0][1] == res_b[0][1]
    assert res_a[0][2] == res_b[0][2]


@pytest.mark.regression
def test_file_handling(fitted_tree, tmp_root_dir):
    x = load_iris().data
    est = Estimator(fitted_tree, language='java')

    # Delete generated files:
    tmp_dir_delete = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'file_handling_delete'),
        ],
    )
    est.make(
        x[:3], n_jobs=False, directory=tmp_dir_delete, delete_created_files=True
    )
    assert len(os.listdir(str(tmp_dir_delete))) == 0

    # Keep generated files:
    tmp_dir_keep = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'file_handling_keep'),
        ],
    )
    est.make(
        x[:3], n_jobs=False, directory=tmp_dir_keep, delete_created_files=False
    )
    assert len(os.listdir(str(tmp_dir_keep))) == 2


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
def test_make_in_directory_with_special_characters(
    fitted_tree, tmp_root_dir: Path, language: str, template: str
):
    """Test paths with spaces and shell characters in the system calls."""
    x = load_iris().data
    directory = tmp_root_dir / f"{language} {template} (a&b) $x 'q';"
    directory.mkdir(parents=True)
    est = Estimator(fitted_tree, language=language, template=template)
    y_pred, _ = est.make(x[:5], n_jobs=False, directory=directory)
    np.testing.assert_array_equal(y_pred, fitted_tree.predict(x[:5]))


@pytest.mark.parametrize('method', ['make', 'test'])
@pytest.mark.parametrize('platform', ['win32', 'linux', 'darwin'])
def test_warning_on_windows(
    fitted_tree, monkeypatch: pytest.MonkeyPatch, method: str, platform: str
):
    """Test the warning of `make()` and `test()` on Windows (#77)."""
    monkeypatch.setattr(ESTIMATOR_MODULE, 'platform', platform)
    monkeypatch.setenv('PATH', '')  # stop at the missing dependencies
    est = Estimator(fitted_tree, language='c')
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        with pytest.raises(RuntimeError, match='Required dependency'):
            getattr(est, method)(load_iris().data[:3])
    messages = [str(r.message) for r in records if r.category is UserWarning]
    is_windows = platform == 'win32'
    assert any('Windows' in m for m in messages) is is_windows


@pytest.mark.parametrize(
    'language, template',
    [('c', 'attached'), ('go', 'exported'), ('java', 'exported')],
)
def test_system_calls_on_windows(
    fitted_tree,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    template: str,
):
    """Test the commands of the system calls on Windows (#77)."""
    tmp_path = tmp_path.resolve()  # like the paths of `make()`
    monkeypatch.setattr(ESTIMATOR_MODULE, 'platform', 'win32')
    monkeypatch.setattr(ESTIMATOR_MODULE, 'pathsep', ';')
    monkeypatch.setenv('SKLEARN_PORTER_PYTEST_GSON_PATH', 'gson.jar')

    compile_cmds, execute_cmds, inputs = [], [], []

    def check_output(cmd: list[str], **kwargs) -> str:
        compile_cmds.append(cmd)
        if language in ('c', 'go'):
            Path(cmd[cmd.index('-o') + 1]).touch()  # the binary
        return ''

    def run(cmd: list[str], **kwargs) -> CompletedProcess:
        execute_cmds.append(cmd)
        inputs.append(kwargs.get('input'))
        return CompletedProcess(cmd, 0, stdout='{"predict": 0}\n', stderr='')

    monkeypatch.setattr(ESTIMATOR_MODULE, 'check_output', check_output)
    monkeypatch.setattr(ESTIMATOR_MODULE, 'run', run)

    est = Estimator(fitted_tree, language=language, template=template)
    with pytest.warns(UserWarning, match='Windows'):
        est.make(
            [[1.0, 2.0, 3.0, 4.0]], check_dependencies=False, directory=tmp_path
        )

    assert len(compile_cmds) == len(execute_cmds) == 1
    compile_cmd, execute_cmd = compile_cmds[0], execute_cmds[0]
    # The features are passed to the standard input:
    assert inputs == ['1.0 2.0 3.0 4.0\n']
    if language in ('c', 'go'):
        binary = str(tmp_path / 'DecisionTreeClassifier.exe')
        assert compile_cmd[compile_cmd.index('-o') + 1] == binary
        assert execute_cmd[0] == binary
    else:
        class_path = f'{tmp_path};gson.jar'
        assert compile_cmd[:3] == ['javac', '-cp', class_path]
        assert execute_cmd[:4] == [
            'java',
            '-cp',
            class_path,
            'DecisionTreeClassifier',
        ]
    if template == 'exported':
        json_path = str(tmp_path / 'DecisionTreeClassifier.json')
        assert execute_cmd[-1] == json_path


@pytest.mark.parametrize(
    'n_samples, n_jobs, n_batches',
    [(3, 4, 1), (250, False, 1), (250, 1, 1), (250, 4, 2), (1000, 4, 4)],
)
def test_make_in_batches(
    fitted_tree,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_samples: int,
    n_jobs: bool | int,
    n_batches: int,
):
    """Test the batches of samples which are passed to the processes."""
    inputs = []

    def run(cmd: list[str], **kwargs) -> CompletedProcess:
        inputs.append(kwargs['input'])
        # Predict the index of each sample (its first feature is 4 * index):
        indices = [
            int(float(e.split()[0])) // 4 for e in kwargs['input'].splitlines()
        ]
        stdout = ''.join(
            f'{{"predict": {i}, "predict_proba": [1, 0, 0]}}\n' for i in indices
        )
        return CompletedProcess(cmd, 0, stdout=stdout, stderr='')

    monkeypatch.setattr(ESTIMATOR_MODULE, 'run', run)

    x = np.arange(n_samples * 4, dtype=float).reshape(n_samples, 4)
    est = Estimator(fitted_tree, language='js')
    y_pred, y_proba = est.make(
        x, n_jobs=n_jobs, check_dependencies=False, directory=tmp_path
    )
    assert len(inputs) == n_batches
    # The batches contain all samples, one sample per line (the threads call
    # the processes in any order):
    inputs.sort(key=lambda batch: float(batch.split()[0]))
    lines = ''.join(inputs).splitlines()
    assert lines == [' '.join(map(str, e)) for e in x.tolist()]
    # The predictions keep the order of the samples:
    np.testing.assert_array_equal(y_pred, np.arange(n_samples))
    assert y_proba.shape == (n_samples, 3)


def test_make_with_missing_results(
    fitted_tree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test the failure if the ported estimator returns too few results."""

    def run(cmd: list[str], **kwargs) -> CompletedProcess:
        return CompletedProcess(cmd, 0, stdout='{"predict": 0}\n', stderr='')

    monkeypatch.setattr(ESTIMATOR_MODULE, 'run', run)
    monkeypatch.setattr(ESTIMATOR_MODULE, 'sleep', lambda _: None)
    est = Estimator(fitted_tree, language='js')
    with pytest.raises(exception.ExecutionFailed, match='1 results of 3'):
        est.make(
            load_iris().data[:3], check_dependencies=False, directory=tmp_path
        )


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
def test_make_with_samples_as_arguments(
    fitted_tree,
    tmp_root_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    template: str,
):
    """Test the features of one sample as arguments of the ported estimator."""
    cmds = []

    def run_and_record(cmd: list[str], **kwargs) -> CompletedProcess:
        cmds.append(cmd)
        return run(cmd, **kwargs)

    monkeypatch.setattr(ESTIMATOR_MODULE, 'run', run_and_record)

    x = load_iris().data[::15]
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'make_with_samples_as_arguments'),
            ('language', language),
            ('template', template),
        ],
    )
    est = Estimator(fitted_tree, language=language, template=template)
    y_pred, y_proba = est.make(x, directory=tmp_dir, delete_created_files=False)
    assert len(cmds) == 1  # one process predicts all samples
    cmd = cmds[0]

    for features, pred, proba in zip(x.tolist(), y_pred, y_proba, strict=True):
        args = list(map(str, features))
        proc = run(
            cmd + args, capture_output=True, encoding='utf-8', check=True
        )
        out = json.loads(proc.stdout)
        assert out['predict'] == pred
        np.testing.assert_allclose(out['predict_proba'], proba, atol=1e-6)

    # Empty lines and additional whitespace between the features:
    samples = '\n  ' + '  '.join(map(str, x[0])) + ' \t\n\n'
    proc = run(cmd, input=samples, capture_output=True, encoding='utf-8')
    assert proc.returncode == 0
    outs = [json.loads(line) for line in proc.stdout.splitlines() if line]
    assert [out['predict'] for out in outs] == [y_pred[0]]

    # A sample with the wrong number of features:
    proc = run(cmd, input='1.0 2.0\n', capture_output=True, encoding='utf-8')
    assert proc.returncode != 0


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('candidate', CANDIDATES, ids=lambda x: x.name)
@pytest.mark.parametrize('binary', [True, False], ids=['binary', 'multi'])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_text_output(
    tmp_root_dir: Path,
    candidate: Candidate,
    binary: bool,
    language: str,
    template: str,
):
    """Test the text output of the ported estimators (`make()` uses JSON)."""
    orig_est = candidate.create()
    is_classifier = candidate in CLASSIFIERS
    if not is_classifier and binary:
        pytest.skip('Skip the binary variant of regressors')
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    params = orig_est.get_params()
    params = {
        k: v
        for k, v in dict(n_estimators=3, max_depth=3, max_iter=20).items()
        if k in params
    }
    orig_est.set_params(**params)
    if is_classifier:
        x, y = load_iris(return_X_y=True)
        if binary:
            y = (y > 0).astype(int)
    else:
        x, y = load_diabetes(return_X_y=True)
    orig_est.fit(X=x, y=y)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'text_output'),
            ('candidate', candidate.name),
            ('binary', str(binary)),
            ('language', language),
            ('template', template),
        ],
    )
    est = Estimator(orig_est, language=language, template=template)
    est._estimator.placeholders['is_test'] = False  # set by pytest
    out = est.save(directory=tmp_dir)
    if isinstance(out, tuple):  # the source code and the model data
        src_path, data_args = Path(out[0]), [str(out[1])]
    else:
        src_path, data_args = Path(out), []

    lang = enum.Language[language.upper()].value
    class_paths = [str(tmp_dir), os.environ['SKLEARN_PORTER_PYTEST_GSON_PATH']]
    cmd_args = dict(
        src_path=str(src_path),
        dest_path=str(tmp_dir / src_path.stem),
        dest_dir=str(tmp_dir),
        class_path=os.pathsep.join(class_paths),
        class_name=src_path.stem,
    )
    if lang.CMD_COMPILE:
        cmd = [arg.format(**cmd_args) for arg in lang.CMD_COMPILE]
        proc = run(cmd, capture_output=True, encoding='utf-8')
        if 'code too large' in proc.stdout + proc.stderr:
            pytest.skip('Skip too large code (Java)')
        assert proc.returncode == 0, proc.stdout + proc.stderr
    cmd = [arg.format(**cmd_args) for arg in lang.CMD_EXECUTE] + data_args

    # The output of several samples is the output of each sample:
    samples = [list(map(str, e)) for e in x[::50]]
    outputs = []
    for args in samples:
        proc = run(cmd + args, capture_output=True, encoding='utf-8')
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() and not proc.stderr
        # e.g. the warnings and deprecations of PHP (see `strict_php`):
        assert not any(w in proc.stdout for w in ('Warning:', 'Deprecated:'))
        outputs.append(proc.stdout)
    batch = ''.join(' '.join(args) + '\n' for args in samples)
    proc = run(cmd, input=batch, capture_output=True, encoding='utf-8')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ''.join(outputs)


def test_missing_compiler(fitted_tree, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PATH', '')
    est = Estimator(fitted_tree, language='c')
    with pytest.raises(exception.CompilationFailed, match='gcc'):
        est.make(load_iris().data[:3], check_dependencies=False)


def test_missing_interpreter(fitted_tree, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PATH', '')
    est = Estimator(fitted_tree, language='js')
    with pytest.raises(exception.ExecutionFailed, match='FileNotFoundError'):
        est.make(load_iris().data[:3], check_dependencies=False, n_jobs=False)


def test_list_of_regressors():
    """Test and compare list of regressors."""
    regressors = sorted(
        [
            r.__name__
            for r in [
                DecisionTreeRegressor,
                ElasticNet,
                ExtraTreesRegressor,
                Lasso,
                LinearRegression,
                MLPRegressor,
                RandomForestRegressor,
                Ridge,
            ]
        ]
    )
    result = sorted([r.__name__ for r in Estimator.regressors()])
    assert regressors == result


def test_list_of_outlier_detectors():
    """Test and compare list of outlier detectors."""
    result = [e.__name__ for e in Estimator.outlier_detectors()]
    assert result == ['IsolationForest']
    assert Estimator.estimators() == (
        Estimator.classifiers()
        + Estimator.regressors()
        + Estimator.outlier_detectors()
    )


def test_list_of_classifiers():
    """Test and compare list of classifiers."""
    classifiers = [
        AdaBoostClassifier,
        BernoulliNB,
        ComplementNB,
        DecisionTreeClassifier,
        ExtraTreesClassifier,
        GaussianNB,
        KNeighborsClassifier,
        LinearSVC,
        LogisticRegression,
        MultinomialNB,
        NuSVC,
        Perceptron,
        RandomForestClassifier,
        RidgeClassifier,
        SGDClassifier,
        SVC,
        MLPClassifier,
    ]
    classifiers = sorted([c.__name__ for c in classifiers])
    result = sorted([c.__name__ for c in Estimator.classifiers()])
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


def test_estimator_extraction_from_pipeline():
    """Test the extraction of an estimator from a pipeline."""
    pipeline = Pipeline([('SVM', SVC())])
    pipeline.fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    est = Estimator(pipeline)
    assert isinstance(est.estimator, SVC)


def test_unfitted_estimator_in_pipeline():
    """Test the extraction of an estimator from a pipeline."""
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
def test_invalid_params_on_port_method(candidate: Candidate, params: tuple):
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
def test_invalid_params_on_save_method(candidate: Candidate, params: tuple):
    """Test initialization with valid base estimator."""
    est = candidate.create().fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    with pytest.raises(params[0]):
        Estimator(est).save(**params[1])


def test_extraction_from_grid_search_cv():
    """Test the extraction from an optimizer."""

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
        X=[[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]], y=[1, 2, 3, 1, 2, 3]
    )
    est = Estimator(search)
    assert isinstance(est.estimator, SVC)


def test_extraction_from_randomized_search_cv():
    """Test the extraction from an optimizer."""

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
        X=[[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]], y=[1, 2, 3, 1, 2, 3]
    )
    est = Estimator(search)
    assert isinstance(est.estimator, SVC)


@pytest.mark.parametrize('orig_est', [SVC, NuSVC], ids=lambda x: x.__name__)
@pytest.mark.parametrize('gamma', ['scale', 'auto'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
def test_computed_gamma_of_svm(orig_est: type, gamma: str, language: str):
    """Test the use of the computed gamma value instead of `scale` or `auto`."""
    # The unscaled iris features (X.var() != 1) lead to different values:
    x, y = load_iris(return_X_y=True)
    orig_est = orig_est(kernel='rbf', gamma=gamma).fit(X=x, y=y)
    expected = orig_est._gamma
    assert expected == (
        1.0 / (x.shape[1] * x.var()) if gamma == 'scale' else 1.0 / x.shape[1]
    )

    est = Estimator(orig_est, language=language, template='attached')
    assert est._estimator.model_data['gamma'] == expected

    output = est.port()
    assert est.converter(expected) in output
    assert gamma not in output


@pytest.mark.parametrize('orig_est', [SVC, NuSVC], ids=lambda x: x.__name__)
def test_svm_with_break_ties(orig_est: type):
    """Test the rejection of `break_ties=True`, which isn't ported."""
    x, y = load_iris(return_X_y=True)
    orig_est = orig_est(break_ties=True).fit(X=x, y=y)
    with pytest.raises(exception.NotSupportedYetError, match='break_ties'):
        Estimator(orig_est)


@pytest.mark.parametrize('orig_est', [SVC, NuSVC], ids=lambda x: x.__name__)
def test_svm_with_precomputed_kernel(orig_est: type):
    """Test the rejection of the kernel `precomputed`, which isn't ported."""
    x, y = load_iris(return_X_y=True)
    orig_est = orig_est(kernel='precomputed').fit(X=x @ x.T, y=y)
    with pytest.raises(exception.NotSupportedYetError, match='precomputed'):
        Estimator(orig_est)


@pytest.mark.parametrize('orig_est', [SVC, NuSVC], ids=lambda x: x.__name__)
def test_svm_fitted_on_sparse_data(orig_est: type):
    """Test the model data of a SVM which was fitted on sparse data."""
    x, y = load_iris(return_X_y=True)
    sparse_fit = orig_est().fit(X=csr_matrix(x), y=y)
    dense_data = Estimator(orig_est().fit(X=x, y=y))._estimator.model_data
    sparse_est = Estimator(sparse_fit)._estimator
    # The variance of `gamma='scale'` is computed differently for sparse data:
    assert sparse_est.model_data.pop('gamma') == sparse_fit._gamma
    dense_data.pop('gamma')
    assert sparse_est.model_data == dense_data
    assert sparse_est.meta_info['n_features'] == x.shape[1]


@pytest.mark.parametrize('binary', [True, False], ids=['binary', 'multi'])
def test_linear_svc_without_intercept(binary: bool):
    """Test a LinearSVC without intercept, which is stored as scalar `0.0`."""
    x, y = load_iris(return_X_y=True)
    if binary:
        y = (y > 0).astype(int)
    orig_est = LinearSVC(fit_intercept=False).fit(X=x, y=y)
    assert orig_est.intercept_ == 0.0
    model_data = Estimator(orig_est)._estimator.model_data
    assert model_data['inters'] == (0.0 if binary else [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    'orig_est',
    [
        DecisionTreeClassifier(random_state=0),
        DecisionTreeRegressor(random_state=0),
        RandomForestClassifier(n_estimators=3, random_state=0),
        RandomForestRegressor(n_estimators=3, random_state=0),
        ExtraTreesClassifier(n_estimators=3, random_state=0),
        ExtraTreesRegressor(n_estimators=3, random_state=0),
    ],
    ids=lambda x: type(x).__name__,
)
def test_tree_routing_of_missing_values(orig_est):
    """Test the routing of missing values (NaN) in the ported trees."""
    x, y = load_iris(return_X_y=True)
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)
    model_data = est._estimator.model_data
    trees = getattr(orig_est, 'estimators_', [orig_est])
    data = model_data.get('estimators', [model_data])
    n_checks = 0
    for tree, tree_data in zip(trees, data, strict=True):
        # The model data contains only the splits:
        is_split = tree.tree_.children_left != -1
        expected = tree.tree_.missing_go_to_left[is_split].tolist()
        assert tree_data['missing_go_to_left'] == expected
        n_checks += sum(expected)
    assert n_checks > 0
    # Only the splits which send missing values to the left child check
    # the features for missing values:
    assert est.port(language='c', template='combined').count('isnan(') == (
        n_checks
    )
    assert '"math"' in est.port(language='go', template='combined')


def test_tree_without_routing_of_missing_values_to_the_left():
    """Test a tree which sends all missing values to the right child."""
    x, y = load_iris(return_X_y=True)
    orig_est = DecisionTreeRegressor(max_depth=1).fit(X=x, y=y)
    assert not orig_est.tree_.missing_go_to_left[0]
    est = Estimator(orig_est)
    assert 'isnan(' not in est.port(language='c', template='combined')
    # Go doesn't compile with unused imports:
    assert '"math"' not in est.port(language='go', template='combined')


@pytest.mark.parametrize(
    'language, literal',
    [
        ('c', 'INFINITY'),
        ('go', 'math.Inf(1)'),
        ('java', 'Double.POSITIVE_INFINITY'),
        ('js', 'Infinity'),
        ('php', 'INF'),
        ('ruby', 'Float::INFINITY'),
    ],
)
@pytest.mark.parametrize(
    'orig_est',
    [
        DecisionTreeClassifier(random_state=0),
        RandomForestRegressor(n_estimators=3, random_state=0),
    ],
    ids=lambda x: type(x).__name__,
)
def test_tree_with_infinite_thresholds(orig_est, language: str, literal: str):
    """Test the infinite thresholds of trees fitted with missing values."""
    x, y = load_iris(return_X_y=True)
    x[np.random.RandomState(0).rand(*x.shape) < 0.2] = np.nan
    orig_est.fit(X=x, y=y)
    trees = getattr(orig_est, 'estimators_', [orig_est])
    assert any(np.isinf(t.tree_.threshold).any() for t in trees)
    est = Estimator(orig_est, language=language)
    assert literal in est.port(template='combined')
    _, model_data = est.port(template='exported')
    assert 'Infinity' not in model_data
    assert '1e999' in model_data
    if language == 'go':
        # Go can't parse `1e999` with encoding/json by default:
        out, _ = est.port(template='exported')
        assert 'UnmarshalJSON' in out


def test_tree_without_infinite_thresholds_in_go():
    """Test that the Go code of common trees doesn't change."""
    x, y = load_iris(return_X_y=True)
    orig_est = DecisionTreeClassifier(random_state=0).fit(X=x, y=y)
    out, model_data = Estimator(orig_est, language='go').port(
        template='exported'
    )
    assert 'UnmarshalJSON' not in out
    assert '1e999' not in model_data


def test_bernoulli_nb_with_infinite_log_probabilities():
    """Test the rejection of feature probabilities of 0 or 1."""
    x, y = load_iris(return_X_y=True)
    x = (x > x.mean(axis=0)).astype(float)
    orig_est = BernoulliNB(alpha=0.0).fit(X=x, y=y)
    assert not np.isfinite(orig_est.feature_log_prob_).all()
    with pytest.raises(exception.NotSupportedYetError, match='alpha'):
        Estimator(orig_est)


@pytest.mark.parametrize('clazz', [MultinomialNB, ComplementNB])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')  # log(0)
def test_naive_bayes_with_infinite_log_probabilities(clazz: type):
    """Test the rejection of feature counts of 0 without smoothing."""
    x = np.array([[1, 0, 1], [2, 0, 1], [0, 3, 1], [0, 1, 2]])
    orig_est = clazz(alpha=0.0).fit(X=x, y=[0, 0, 1, 1])
    assert not np.isfinite(orig_est.feature_log_prob_).all()
    with pytest.raises(exception.NotSupportedYetError, match='alpha'):
        Estimator(orig_est)
    # With `force_alpha=False` scikit-learn smoothes with `alpha=1e-10`:
    orig_est = clazz(alpha=0.0, force_alpha=False)
    with pytest.warns(UserWarning, match='alpha too small'):
        orig_est.fit(X=x, y=[0, 0, 1, 1])
    model_data = Estimator(orig_est)._estimator.model_data
    assert model_data['probs'] == orig_est.feature_log_prob_.tolist()


@pytest.mark.filterwarnings('ignore::RuntimeWarning')  # log(0)
def test_multinomial_nb_with_class_probabilities_of_zero():
    """Test the rejection of classes with a probability of 0."""
    x, y = load_iris(return_X_y=True)
    orig_est = MultinomialNB(class_prior=[0.0, 0.5, 0.5]).fit(X=x, y=y)
    with pytest.raises(exception.NotSupportedYetError, match='class_prior'):
        Estimator(orig_est)
    # Classes without samples (e.g. in the first batch of `partial_fit`):
    orig_est = MultinomialNB().partial_fit(X=x[:50], y=y[:50], classes=[0, 1])
    with pytest.raises(exception.NotSupportedYetError, match='class_prior'):
        Estimator(orig_est)


def test_multinomial_nb_model_data():
    """Test the model data of a MultinomialNB."""
    x, y = load_iris(return_X_y=True)
    orig_est = MultinomialNB(class_prior=[0.2, 0.3, 0.5]).fit(X=x, y=y)
    model_data = Estimator(orig_est)._estimator.model_data
    assert model_data['priors'] == orig_est.class_log_prior_.tolist()
    assert model_data['probs'] == orig_est.feature_log_prob_.tolist()


@pytest.mark.parametrize('n_classes', [1, 2, 3])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')  # log(0)
def test_complement_nb_model_data(n_classes: int):
    """Test the class priors of a ComplementNB, which are only used with a
    single class like in scikit-learn."""
    x, y = load_iris(return_X_y=True)
    y = np.minimum(y, n_classes - 1)
    class_prior = np.zeros(n_classes)
    class_prior[-1] = 0.5  # e.g. infinite log priors, which are ignored
    orig_est = ComplementNB(class_prior=class_prior).fit(X=x, y=y)
    model_data = Estimator(orig_est)._estimator.model_data
    priors = orig_est.class_log_prior_ if n_classes == 1 else [0.0] * n_classes
    assert model_data['priors'] == list(priors)
    assert model_data['probs'] == orig_est.feature_log_prob_.tolist()
    # The ported joint log likelihoods (of the ported model data):
    jll = x @ np.array(model_data['probs']).T + model_data['priors']
    np.testing.assert_allclose(jll, orig_est.predict_joint_log_proba(x))


SPARSIFIABLE_CLASSIFIERS = [
    LinearSVC(random_state=0),
    LogisticRegression(max_iter=1000),
    SGDClassifier(random_state=0),
    Perceptron(random_state=0),
]


@pytest.mark.parametrize(
    'orig_est', SPARSIFIABLE_CLASSIFIERS, ids=lambda x: type(x).__name__
)
@pytest.mark.parametrize('binary', [True, False], ids=['binary', 'multi'])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_sparsified_linear_classifier(orig_est, binary: bool):
    """Test the model data of a classifier with sparse coefficients."""
    x, y = load_iris(return_X_y=True)
    if binary:
        y = (y > 0).astype(int)
    orig_est = clone(orig_est).fit(X=x, y=y)
    dense_est = Estimator(orig_est)._estimator
    orig_est.sparsify()
    est = Estimator(orig_est)
    sparse_est = est._estimator
    assert sparse_est.model_data == dense_est.model_data
    assert sparse_est.meta_info == dense_est.meta_info
    # The passed estimator remains unchanged:
    assert est.estimator is orig_est
    assert issparse(orig_est.coef_)


def test_knn_fitted_on_sparse_data():
    """Test the model data of a KNeighborsClassifier fitted on sparse data."""
    x, y = load_iris(return_X_y=True)
    dense_est = Estimator(KNeighborsClassifier().fit(X=x, y=y))._estimator
    sparse_fit = KNeighborsClassifier().fit(X=csr_matrix(x), y=y)
    est = Estimator(sparse_fit)
    sparse_est = est._estimator
    assert sparse_est.model_data == dense_est.model_data
    assert sparse_est.meta_info == dense_est.meta_info
    # The passed estimator remains unchanged:
    assert est.estimator is sparse_fit
    assert issparse(sparse_fit._fit_X)


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize(
    'orig_est',
    [*SPARSIFIABLE_CLASSIFIERS, KNeighborsClassifier()],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize('binary', [True, False], ids=['binary', 'multi'])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_classifier_with_sparse_data(
    orig_est, binary: bool, language: str, template: str
):
    """Test the port of classifiers with sparse coefficients or samples."""
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = load_iris(return_X_y=True)
    if binary:
        y = (y > 0).astype(int)
    if isinstance(orig_est, KNeighborsClassifier):
        orig_est = clone(orig_est).fit(X=csr_matrix(x), y=y)
    else:
        orig_est = clone(orig_est).fit(X=x, y=y).sparsify()
    est = Estimator(orig_est, language=language, template=template)
    # The ported estimator also accepts sparse samples:
    x_test = csr_matrix(x[::5])
    y_pred, _ = est.make(x_test, n_jobs=False)
    np.testing.assert_array_equal(y_pred, orig_est.predict(x_test))
    assert est.test(x_test, n_jobs=False) == 1.0


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize(
    'orig_est',
    [
        DecisionTreeClassifier(max_depth=3, random_state=0),
        RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0),
        ExtraTreesClassifier(n_estimators=3, max_depth=3, random_state=0),
        KNeighborsClassifier(),
    ],
    ids=lambda x: type(x).__name__,
)
def test_multi_output_classifier(orig_est, language: str, template: str):
    """Test the port of classifiers fitted on a target with several outputs."""
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = load_iris(return_X_y=True)
    y = np.c_[y, x[:, 0] > 5.8, np.digitize(x[:, 2], [2, 4, 5.5])]
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est, language=language, template=template)
    meta_info = est._estimator.meta_info
    assert meta_info['is_multi_output']
    assert meta_info['output_classes'] == [3, 2, 4]
    assert meta_info['n_classes'] == 9
    output = est.port()
    # The multi-output variant predicts one class per output:
    est._estimator.meta_info['is_multi_output'] = False
    assert est.port() != output


def test_multi_output_k_neighbors_classifier_targets():
    """Test the concatenated classes of the samples of a KNN classifier."""
    x, y = load_iris(return_X_y=True)
    y = np.c_[y, x[:, 0] > 5.8]
    orig_est = KNeighborsClassifier().fit(X=x, y=y)
    model_data = Estimator(orig_est)._estimator.model_data
    # The classes of the second output start at the offset 3:
    expected = np.c_[y[:, 0], y[:, 1] + 3].ravel()
    np.testing.assert_array_equal(model_data['y'], expected)
    assert model_data['n'] == 3 + 2


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize(
    'orig_est',
    [RidgeClassifier(), MLPClassifier(max_iter=20, random_state=0)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_multilabel_classifier(
    orig_est: RidgeClassifier | MLPClassifier, language: str, template: str
):
    """Test the port of classifiers fitted on a label indicator matrix."""
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = make_multilabel_classification(n_classes=3, random_state=0)
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est, language=language, template=template)
    assert est._estimator.meta_info['is_multilabel']
    output = est.port()
    # The multilabel variant predicts one binary value per label:
    est._estimator.meta_info['is_multilabel'] = False
    assert est.port() != output


@pytest.mark.parametrize(
    ('binarize', 'threshold'),
    [(0.0, 0.0), (3, 3.0), (None, 0.0)],
    ids=['zero', 'int', 'none'],
)
def test_bernoulli_nb_with_binarize(binarize, threshold: float):
    """Test the threshold to binarize the features of a BernoulliNB."""
    x, y = load_iris(return_X_y=True)
    orig_est = BernoulliNB(binarize=binarize).fit(X=x, y=y)
    model_data = Estimator(orig_est)._estimator.model_data
    assert model_data['binarize'] == (binarize is not None)
    assert model_data['threshold'] == threshold
    assert isinstance(model_data['threshold'], float)


@pytest.mark.parametrize(
    ('params', 'metric', 'power'),
    [
        (dict(), 'minkowski', 2),
        (dict(metric='euclidean'), 'minkowski', 2),
        (dict(metric='l2'), 'minkowski', 2),
        (dict(p=1), 'minkowski', 1),
        (dict(metric='manhattan'), 'minkowski', 1),
        (dict(metric='cityblock'), 'minkowski', 1),
        (dict(metric='l1'), 'minkowski', 1),
        (dict(p=3), 'minkowski', 3),
        (dict(p=0.5), 'minkowski', 0.5),
        (dict(metric_params={'p': 3}), 'minkowski', 3),
        (dict(metric='chebyshev'), 'chebyshev', 2),
        (dict(metric='infinity'), 'chebyshev', 2),
        (dict(p=np.inf), 'chebyshev', 2),
        (dict(metric='cosine'), 'cosine', 2),
        (dict(metric='hamming'), 'hamming', 2),
    ],
    ids=lambda x: str(x).replace(' ', ''),
)
def test_k_neighbors_classifier_with_supported_metric(
    params: dict, metric: str, power: float
):
    """Test the metric and the power of the ported distance."""
    x, y = load_iris(return_X_y=True)
    orig_est = KNeighborsClassifier(**params).fit(X=x, y=y)
    model_data = Estimator(orig_est)._estimator.model_data
    assert model_data['metric'] == metric
    assert model_data['power'] == power


@pytest.mark.parametrize(
    ('params', 'match'),
    [
        (dict(metric='canberra'), 'canberra'),
        (dict(metric=lambda a, b: np.abs(a - b).sum()), '<lambda>'),
        (dict(metric_params={'w': [1.0, 2.0, 3.0, 4.0]}), 'weighted'),
    ],
    ids=['canberra', 'callable', 'weighted'],
)
def test_k_neighbors_classifier_with_unsupported_metric(
    params: dict, match: str
):
    """Test the rejection of metrics which aren't ported."""
    x, y = load_iris(return_X_y=True)
    orig_est = KNeighborsClassifier(**params).fit(X=x, y=y)
    with pytest.raises(exception.NotSupportedYetError, match=match):
        Estimator(orig_est)


@pytest.mark.parametrize('weights', ['uniform', 'distance'])
def test_k_neighbors_classifier_with_supported_weights(weights: str):
    """Test the weights of the votes of the ported neighbors."""
    x, y = load_iris(return_X_y=True)
    orig_est = KNeighborsClassifier(weights=weights).fit(X=x, y=y)
    assert Estimator(orig_est)._estimator.meta_info['weights'] == weights


def test_k_neighbors_classifier_with_unsupported_weights():
    """Test the rejection of weights which aren't ported (callables)."""
    x, y = load_iris(return_X_y=True)
    orig_est = KNeighborsClassifier(weights=lambda d: d**2).fit(X=x, y=y)
    with pytest.raises(exception.NotSupportedYetError, match='weights'):
        Estimator(orig_est)


@pytest.mark.regression
@pytest.mark.parametrize(
    'x',
    [
        [5.1, 3.5, 1.4, 0.2],
        np.array([5.1, 3.5, 1.4, 0.2]),
        [[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]],
        np.array([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]]),
        [np.array([5.1, 3.5, 1.4, 0.2]), np.array([5.1, 3.5, 1.4, 0.2])],
    ],
    ids=[  # fmt: off
        f'{type([5.1, 3.5, 1.4, 0.2]).__qualname__}_{type([5.1, 3.5, 1.4, 0.2][0]).__qualname__}',
        f'{type(np.array([5.1, 3.5, 1.4, 0.2])).__qualname__}_{type(np.array([5.1, 3.5, 1.4, 0.2])[0]).__qualname__}',
        f'{type([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]]).__qualname__}_{type([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]][0]).__qualname__}',
        f'{type(np.array([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]])).__qualname__}_{type(np.array([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]])[0]).__qualname__}',
        f'{type([np.array([5.1, 3.5, 1.4, 0.2]), np.array([5.1, 3.5, 1.4, 0.2])]).__qualname__}_{type([np.array([5.1, 3.5, 1.4, 0.2]), np.array([5.1, 3.5, 1.4, 0.2])][0]).__qualname__}',
    ],  # fmt: on
)
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
def test_make_the_inputs_outputs(
    tmp_root_dir: Path,
    x,
    fitted_tree: DecisionTreeClassifier,
    template: str,
    language: str,
):
    if not can(fitted_tree, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'make_the_inputs_outputs'),
            ('estimator', 'DecisionTreeClassifier'),
            ('language', language),
            ('template', template),
        ],
    )

    # Estimator:
    est = Estimator(fitted_tree)
    out = est.make(
        x,
        language=language,
        template=template,
        directory=tmp_dir,
        n_jobs=False,
        delete_created_files=False,
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
@pytest.mark.parametrize('candidate', CANDIDATES, ids=lambda x: x.name)
def test_port_all_combinations(
    candidate: Candidate, language: str, template: str
):
    """Port every supported combination without compiling the output."""
    if not can(candidate.clazz(), language, template):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = load_iris(return_X_y=True)
    orig_est = candidate.create()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        orig_est.fit(X=x, y=y)
    out = Estimator(orig_est, language=language, template=template).port()
    outputs = out if isinstance(out, tuple) else (out,)
    assert all(isinstance(o, str) and o for o in outputs)


@pytest.mark.regression
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

    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    if isinstance(orig_est, KNeighborsClassifier):
        test_x = drop_neighbor_ties(orig_est, test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'regressions_of_classifiers'),
            ('estimator', candidate.name),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )

    try:
        y_pred, y_proba = est.make(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except exception.CodeTooLarge:
        msg = (
            'The code too large for the combination: '
            f'estimator: {candidate.name}, language: {language}, template: {template}, dataset: {dataset.name}'
        )
        warnings.warn(msg, stacklevel=2)
    except exception.TooManyConstants:
        msg = (
            'The code has too many constants and can not be compiled: '
            f'estimator: {candidate.name}, language: {language}, template: {template}, dataset: {dataset.name}'
        )
        warnings.warn(msg, stacklevel=2)
    except Exception as e:
        pytest.fail('Unexpected exception ... ' + str(e))
    else:
        np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
        if can(orig_est, language, template, 'predict_proba'):
            np.testing.assert_allclose(
                y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-5
            )


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', REGRESSION_DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('candidate', REGRESSORS, ids=lambda x: x.name)
def test_regressions_of_regressors(
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'regressions_of_regressors'),
            ('estimator', candidate.name),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )

    try:
        y_pred, _ = est.make(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except (exception.CodeTooLarge, exception.TooManyConstants) as e:
        pytest.skip(str(e))
    np.testing.assert_allclose(
        y_pred.astype(float), orig_est.predict(test_x), rtol=1e-6, atol=1e-8
    )
    score = est.test(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    assert score == 1.0


# The labels of the classes of the tests, the strings contain the quotes,
# escapes, trigraphs, interpolations and characters outside of ASCII of the
# languages:
LABELS = {
    'int': [-1, 3, 7],
    'float': [-1.0, 2.0, 5.0],
    'str': [
        "set'osa",
        'ver"si\\color ??= $x #{y}',
        'vir\tginica \xe4\U0001f600',
    ],
    'binary_int': [-1, 1],
    'binary_bool': [False, True],
}


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('labels', LABELS.values(), ids=list(LABELS))
@pytest.mark.parametrize('candidate', CLASSIFIERS, ids=lambda x: x.name)
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_class_labels(
    tmp_root_dir: Path,
    candidate: Candidate,
    labels: list,
    language: str,
    template: str,
):
    """Test the labels of the classes, which the ported classifiers return."""
    if not can(candidate.clazz(), language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = load_iris(return_X_y=True)
    # Binary labels merge the last classes:
    y = np.array(labels)[np.minimum(y, len(labels) - 1)]
    orig_est = candidate.create().fit(X=x, y=y)
    test_x = x[::10]
    if isinstance(orig_est, KNeighborsClassifier):
        test_x = drop_neighbor_ties(orig_est, test_x)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'class_labels'),
            ('estimator', candidate.name),
            ('labels', str(len(labels)) + type(labels[0]).__name__),
            ('language', language),
            ('template', template),
        ],
    )
    y_pred, _ = Estimator(orig_est).make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    assert y_pred.dtype == orig_est.classes_.dtype
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize(
    'orig_est',
    [
        DecisionTreeClassifier(max_depth=3, random_state=0),
        RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0),
        ExtraTreesClassifier(n_estimators=3, max_depth=3, random_state=0),
        KNeighborsClassifier(),
    ],
    ids=lambda x: type(x).__name__,
)
def test_class_labels_of_multiple_outputs(
    tmp_root_dir: Path, orig_est, language: str, template: str
):
    """Test the labels of the classes of each output."""
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')
    x, y = load_iris(return_X_y=True)
    y = np.c_[
        np.array(LABELS['str'])[y],
        np.where(x[:, 0] > 5.8, 'long', 'short'),
    ]
    orig_est.fit(X=x, y=y)
    test_x = x[::10]
    if isinstance(orig_est, KNeighborsClassifier):
        test_x = drop_neighbor_ties(orig_est, test_x)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'class_labels_of_multiple_outputs'),
            ('estimator', type(orig_est).__name__),
            ('language', language),
            ('template', template),
        ],
    )
    y_pred, _ = Estimator(orig_est).make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('n_outputs', [1, 2])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_mlp_regressor_with_poisson_loss(
    tmp_root_dir: Path, n_outputs: int, language: str, template: str
):
    """Test the output activation `exp` of a MLPRegressor with the loss
    `poisson`."""
    x, y = load_diabetes(return_X_y=True)
    if n_outputs == 2:
        y = np.column_stack((y, y / 2.0 + 1.0))
    orig_est = MLPRegressor(
        loss='poisson', hidden_layer_sizes=(10,), max_iter=300, random_state=0
    ).fit(X=x, y=y)
    assert orig_est.out_activation_ == 'exp'

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'mlp_regressor_with_poisson_loss'),
            ('n_outputs', str(n_outputs)),
            ('language', language),
            ('template', template),
        ],
    )
    test_x = x[:100]
    y_pred, _ = Estimator(orig_est).make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_allclose(
        y_pred.astype(float), orig_est.predict(test_x), rtol=1e-6
    )


@pytest.mark.parametrize(
    'orig_est',
    [
        MLPClassifier(hidden_layer_sizes=(5, 4), max_iter=5, random_state=0),
        MLPRegressor(hidden_layer_sizes=(5, 4), max_iter=5, random_state=0),
    ],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_mlp_layers_from_fitted_weights(orig_est: MLPClassifier | MLPRegressor):
    """Test the sizes of the layers of a MLP whose `hidden_layer_sizes`
    were changed after the fit."""
    x, y = load_iris(return_X_y=True)
    orig_est.fit(X=x, y=y)
    orig_est.set_params(hidden_layer_sizes=(3,))
    n_outputs = 3 if isinstance(orig_est, MLPClassifier) else 1
    model_data = Estimator(orig_est)._estimator.model_data
    assert model_data['layers'] == [5, 4, n_outputs]
    assert all(isinstance(n, int) for n in model_data['layers'])


def test_mlp_regressor_with_unsupported_output_activation():
    """Test the rejection of unknown output activations of a MLPRegressor."""
    x, y = load_diabetes(return_X_y=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        orig_est = MLPRegressor(max_iter=5, random_state=0).fit(X=x, y=y)
    orig_est.out_activation_ = 'tanh'
    with pytest.raises(exception.NotSupportedYetError, match='activation'):
        Estimator(orig_est)


@pytest.mark.parametrize('n_estimators', [3, 12])
@pytest.mark.parametrize(
    'orig_est',
    [
        RandomForestClassifier(n_estimators=6, max_depth=3, random_state=0),
        ExtraTreesClassifier(n_estimators=6, max_depth=3, random_state=0),
    ],
    ids=lambda x: type(x).__name__,
)
def test_forest_n_estimators_from_fitted_trees(
    orig_est: RandomForestClassifier | ExtraTreesClassifier, n_estimators: int
):
    """Test the number of trees of a forest whose `n_estimators` was changed
    after the fit."""
    x, y = load_iris(return_X_y=True)
    orig_est = clone(orig_est).fit(X=x, y=y)
    orig_est.set_params(n_estimators=n_estimators)
    est = Estimator(orig_est)._estimator
    assert est.meta_info['n_estimators'] == 6
    assert len(est.model_data['estimators']) == 6


def _score_isolation_forest(model_data: dict, x: np.ndarray) -> np.ndarray:
    """Sum the path lengths of the trees like the ported IsolationForest."""
    scores = []
    for features in x:
        path_length = 0.0
        for tree in model_data['estimators']:
            node = 0
            while node >= 0:
                feature = features[tree['indices'][node]]
                is_left = feature <= tree['thresholds'][node] or (
                    tree['missing_go_to_left'][node] and np.isnan(feature)
                )
                node = tree['lefts' if is_left else 'rights'][node]
            path_length += tree['values'][-node - 1]
        scores.append(-(2 ** -(path_length / model_data['denominator'])))
    return np.array(scores)


@pytest.mark.parametrize(
    'params',
    [
        dict(),
        dict(contamination=0.1),
        dict(max_features=0.5),
        dict(max_samples=0.5, bootstrap=True),
        dict(max_samples=1),
    ],
    ids=['default', 'contamination', 'max_features', 'bootstrap', 'one_sample'],
)
def test_isolation_forest_model_data(params: dict):
    """Test the model data of an IsolationForest: the path lengths of the
    leaves give exactly the scores of scikit-learn."""
    x, _ = load_iris(return_X_y=True)
    x = x.copy()
    x[::7, 1] = np.nan  # missing values
    orig_est = IsolationForest(n_estimators=10, random_state=0, **params)
    orig_est.fit(x)
    est = Estimator(orig_est, language='js', template='exported')
    model_data = est._estimator.model_data
    assert len(model_data['estimators']) == 10
    assert model_data['offset'] == orig_est.offset_
    test_x = np.vstack((x, x + 0.5))
    scores = _score_isolation_forest(model_data, test_x)
    np.testing.assert_array_equal(scores, orig_est.score_samples(test_x))
    y_pred = np.where(scores - model_data['offset'] < 0, -1, 1)
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    # The template `exported` saves the denominator and the offset too:
    _, exported = est.port()
    assert json.loads(exported) == json.loads(json.dumps(model_data))


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('n_estimators', [3, 12])
@pytest.mark.parametrize(
    'orig_est',
    [
        RandomForestClassifier(n_estimators=6, max_depth=3, random_state=0),
        ExtraTreesClassifier(n_estimators=6, max_depth=3, random_state=0),
    ],
    ids=lambda x: type(x).__name__,
)
def test_forest_with_changed_n_estimators(
    tmp_root_dir: Path,
    orig_est: RandomForestClassifier | ExtraTreesClassifier,
    n_estimators: int,
    language: str,
    template: str,
):
    """Test the predictions of a forest whose `n_estimators` was changed
    after the fit."""
    x, y = load_iris(return_X_y=True)
    orig_est = clone(orig_est).fit(X=x, y=y)
    orig_est.set_params(n_estimators=n_estimators)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'forest_with_changed_n_estimators'),
            ('estimator', type(orig_est).__name__),
            ('n_estimators', str(n_estimators)),
            ('language', language),
            ('template', template),
        ],
    )
    try:
        y_pred, y_proba = Estimator(orig_est).make(
            x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False,
        )
    except (exception.CodeTooLarge, exception.TooManyConstants) as e:
        pytest.skip(str(e))
    np.testing.assert_array_equal(y_pred, orig_est.predict(x))
    np.testing.assert_allclose(
        y_proba, orig_est.predict_proba(x), rtol=1e-6, atol=1e-8
    )


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize(
    'orig_est',
    [
        DecisionTreeClassifier(min_samples_split=1000),
        DecisionTreeRegressor(min_samples_split=1000),
        RandomForestClassifier(
            n_estimators=3, min_samples_split=1000, random_state=0
        ),
        RandomForestRegressor(
            n_estimators=3, min_samples_split=1000, random_state=0
        ),
    ],
    ids=lambda x: type(x).__name__,
)
def test_trees_with_a_single_leaf(
    tmp_root_dir: Path,
    orig_est: BaseEstimator,
    language: str,
    template: str,
):
    """Test trees which consist of a single leaf. Their model data has a
    split whose children are both this leaf (see `get_tree_data()`)."""
    x, y = load_iris(return_X_y=True)
    orig_est = clone(orig_est).fit(X=x, y=y)
    trees = getattr(orig_est, 'estimators_', [orig_est])
    assert all(tree.tree_.node_count == 1 for tree in trees)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'trees_with_a_single_leaf'),
            ('estimator', type(orig_est).__name__),
            ('language', language),
            ('template', template),
        ],
    )
    y_pred, y_proba = Estimator(orig_est).make(
        x[::10],
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(
        y_pred.astype(float), orig_est.predict(x[::10])
    )
    if hasattr(orig_est, 'predict_proba'):
        # Some ported estimators print the probabilities with six decimals:
        np.testing.assert_allclose(
            y_proba, orig_est.predict_proba(x[::10]), rtol=0, atol=1e-5
        )


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('candidate', CANDIDATES, ids=lambda x: x.name)
def test_php_without_errors(
    tmp_root_dir: Path,
    candidate: Candidate,
    template: str,
):
    """Test that the ported PHP code raises no notices, warnings or
    deprecations, e.g. dynamic properties (deprecated since PHP 8.2)."""
    if not can(candidate.clazz(), 'php', template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    x, y = load_iris(return_X_y=True)
    orig_est = candidate.create()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        orig_est.fit(X=x, y=y)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'php_without_errors'),
            ('estimator', candidate.name),
            ('template', template),
        ],
    )
    out = Estimator(orig_est, language='php', template=template).save(
        directory=tmp_dir
    )
    src_path, data_path = out if isinstance(out, tuple) else (out, None)

    # Ignore the `php.ini` and show all errors on stderr:
    cmd = ['php', '-n', '-d', 'error_reporting=-1']
    cmd += ['-d', 'display_errors=stderr', '-f', str(src_path), '--']
    if data_path:
        cmd.append(str(data_path))
    cmd += list(map(str, x[0]))
    proc = run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stderr == ''
    assert proc.stdout


@pytest.mark.parametrize('subcommand', ['show', 'port', 'save'])
def test_cli_subcommand(subcommand: str, joblib_model_path: Path):
    args = [subcommand]
    if subcommand in ['port', 'save']:
        args += [str(joblib_model_path), '--skip-warnings']
    parsed = parse_args(args)
    cmd = args[0]
    assert parsed.cmd == cmd


def test_cli_subcommand_show():
    out = show_main(dict(), silent=True)
    assert 'DecisionTreeClassifier' in out
    assert 'Java' in out
    assert 'JavaScript' in out

    out = show_main(dict(language='c'), silent=True)
    assert 'DecisionTreeClassifier' in out
    assert 'Java' not in out
    assert 'JavaScript' not in out

    with pytest.raises(exception.InvalidLanguageError):
        show_main(dict(language='i_n_v_a_l_i_d'), silent=True)


def test_cli_subcommand_port(joblib_model_path: Path):
    out = port_main(
        dict(model=str(joblib_model_path), skip_warnings=True), silent=True
    )
    assert 'This file is generated by' in out

    out = port_main(
        dict(
            model=str(joblib_model_path),
            langauge='java',
            template='combined',
            skip_warnings=True,
        ),
        silent=True,
    )
    assert 'This file is generated by' in out

    out = port_main(
        dict(language='js', model=str(joblib_model_path), skip_warnings=True),
        silent=True,
    )
    assert 'node' in out and 'require.main === module' in out


def test_cli_subcommand_save(tmp_root_dir: Path, joblib_model_path: Path):
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'cli_subcommand_save'),
            ('case', 'a'),
        ],
    )
    out = save_main(
        dict(
            model=str(joblib_model_path),
            directory=str(tmp_dir),
            skip_warnings=True,
        ),
        silent=True,
    )
    assert 'Saved files' in out

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'cli_subcommand_save'),
            ('case', 'b'),
        ],
    )
    out = save_main(
        dict(
            model=str(joblib_model_path),
            langauge='java',
            template='combined',
            directory=str(tmp_dir),
            skip_warnings=True,
        ),
        silent=True,
    )
    assert 'Saved files' in out


def test_cli_load_model_warnings(
    monkeypatch: pytest.MonkeyPatch, joblib_model_path: Path
):
    """Warnings raised while loading are errors unless they are skipped."""
    load = cli_utils.joblib.load

    def load_with_warning(*args, **kwargs):
        warnings.warn('Loading warning', UserWarning, stacklevel=2)
        return load(*args, **kwargs)

    monkeypatch.setattr(cli_utils.joblib, 'load', load_with_warning)
    with pytest.raises(UserWarning):
        cli_utils.load_model(joblib_model_path)
    mdl = cli_utils.load_model(joblib_model_path, skip_warnings=True)
    assert isinstance(mdl, DecisionTreeClassifier)


def test_cli_load_model_restores_warning_filters(
    monkeypatch: pytest.MonkeyPatch, joblib_model_path: Path
):
    """Warnings raised after loading, e.g. while porting, aren't errors."""
    port = Estimator.port

    def port_with_warning(*args, **kwargs):
        warnings.warn('Porting warning', UserWarning, stacklevel=2)
        return port(*args, **kwargs)

    monkeypatch.setattr(Estimator, 'port', port_with_warning)
    with pytest.warns(UserWarning, match='Porting warning'):
        out = port_main(dict(model=str(joblib_model_path)), silent=True)
    assert 'This file is generated by' in out
