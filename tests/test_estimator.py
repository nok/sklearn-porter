import os
import warnings
from collections.abc import Callable
from pathlib import Path
from subprocess import run

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
from sklearn.naive_bayes import BernoulliNB, GaussianNB
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
    """Test the quoting of the paths in the shell commands."""
    x = load_iris().data
    directory = tmp_root_dir / f"{language} {template} (a&b) $x 'q';"
    directory.mkdir(parents=True)
    est = Estimator(fitted_tree, language=language, template=template)
    y_pred, _ = est.make(x[:5], n_jobs=False, directory=directory)
    np.testing.assert_array_equal(y_pred, fitted_tree.predict(x[:5]))


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
        LogisticRegression,
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
        expected = tree.tree_.missing_go_to_left.tolist()
        assert tree_data['missing_go_to_left'] == expected
        is_split = tree.tree_.children_left != -1
        n_checks += int(tree.tree_.missing_go_to_left[is_split].sum())
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


def test_mlp_regressor_with_unsupported_output_activation():
    """Test the rejection of unknown output activations of a MLPRegressor."""
    x, y = load_diabetes(return_X_y=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        orig_est = MLPRegressor(max_iter=5, random_state=0).fit(X=x, y=y)
    orig_est.out_activation_ = 'tanh'
    with pytest.raises(exception.NotSupportedYetError, match='activation'):
        Estimator(orig_est)


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
