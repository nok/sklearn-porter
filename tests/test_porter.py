import re
import warnings
from pathlib import Path

import pytest

# scikit-learn
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
import sklearn_porter
from sklearn_porter import (
    Estimator,
    Porter,
    Transformer,
    can,
    port,
    save,
    show,
)
from sklearn_porter import Pipeline as PortedPipeline
from sklearn_porter import exceptions as exception
from tests.commons import LANGUAGES


@pytest.fixture(scope='module')
def fitted_tree() -> DecisionTreeClassifier:
    x, y = load_iris(return_X_y=True)
    return DecisionTreeClassifier(random_state=0).fit(x, y)


@pytest.fixture(scope='module')
def fitted_scaler() -> StandardScaler:
    return StandardScaler().fit(load_iris().data)


def test_porter_of_estimator(fitted_tree: DecisionTreeClassifier):
    porter = Porter(fitted_tree, language='java', template='exported')
    assert isinstance(porter.ported, Estimator)
    assert porter.estimator is fitted_tree
    assert porter.language == 'java'
    assert porter.template == 'exported'
    assert porter.class_name == 'DecisionTreeClassifier'
    assert repr(porter) == repr(porter.ported)
    expected = Estimator(fitted_tree, language='java', template='exported')
    assert porter.port() == expected.port()


def test_porter_of_transformer(fitted_scaler: StandardScaler):
    porter = Porter(fitted_scaler, language='c', template='combined')
    assert isinstance(porter.ported, Transformer)
    assert porter.estimator is fitted_scaler
    assert porter.language == 'c'
    assert porter.template == 'combined'
    assert porter.class_name == 'StandardScaler'
    assert repr(porter) == repr(porter.ported)
    expected = Transformer(fitted_scaler, language='c', template='combined')
    assert porter.port() == expected.port()


def test_porter_setters(fitted_scaler: StandardScaler):
    porter = Porter(fitted_scaler)
    porter.language = 'go'
    porter.template = 'exported'
    porter.class_name = 'Scaler'
    porter.converter = repr
    assert porter.ported.language == 'go'
    assert porter.ported.template == 'exported'
    assert porter.ported.class_name == 'Scaler'
    assert porter.ported.converter is repr
    assert 'Scaler' in porter.port()[0]


def test_porter_can(
    fitted_tree: DecisionTreeClassifier, fitted_scaler: StandardScaler
):
    assert Porter(fitted_tree).can('c', 'attached', 'predict_proba')
    assert not Porter(fitted_tree).can('c', 'attached', 'transform')
    assert Porter(fitted_scaler).can('c', 'attached', 'transform')
    assert not Porter(fitted_scaler).can('c', 'attached', 'predict')


def test_porter_save(fitted_scaler: StandardScaler, tmp_path: Path):
    paths = Porter(fitted_scaler, template='exported').save(tmp_path)
    assert paths == (
        str(tmp_path / 'StandardScaler.js'),
        str(tmp_path / 'StandardScaler.json'),
    )


def test_porter_wrong_kind(
    fitted_tree: DecisionTreeClassifier, fitted_scaler: StandardScaler
):
    """`predict` is only available for estimators, `transform` only for
    transformers."""
    with pytest.raises(AttributeError, match='use `predict` instead'):
        Porter(fitted_tree).transform([[1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(AttributeError, match='use `transform` instead'):
        Porter(fitted_scaler).predict([[1.0, 2.0, 3.0, 4.0]])


def test_porter_of_pipeline():
    """A pipeline with supported steps creates a `Pipeline`, with the
    opt-out `ignore_preprocessing` an `Estimator` (of the final estimator)."""
    x, y = load_iris(return_X_y=True)
    pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC())]).fit(
        x, y
    )
    porter = Porter(pipeline, language='go')
    assert isinstance(porter.ported, PortedPipeline)
    assert porter.estimator is pipeline
    assert porter.language == 'go'
    with pytest.raises(AttributeError, match='use `predict` instead'):
        porter.transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        porter = Porter(pipeline, ignore_preprocessing=True)
    assert isinstance(porter.ported, Estimator)
    assert porter.estimator is pipeline[-1]


def test_porter_of_pipeline_with_unsupported_steps():
    """A pipeline with unsupported steps raises the error of `Estimator`."""
    x, y = load_iris(return_X_y=True)
    pipeline = Pipeline([('spline', SplineTransformer()), ('svm', SVC())]).fit(
        x, y
    )
    with pytest.raises(exception.NotSupportedYetError, match='`spline`'):
        Porter(pipeline)


def test_porter_of_unsupported_transformer():
    spline = SplineTransformer().fit(load_iris().data)
    with pytest.raises(
        exception.NotSupportedYetError, match='`SplineTransformer`'
    ):
        Porter(spline)


@pytest.mark.parametrize('obj', [None, object, 0, 'string'])
def test_porter_of_invalid_objects(obj):
    with pytest.raises(ValueError):
        Porter(obj)


def test_functions(
    fitted_tree: DecisionTreeClassifier,
    fitted_scaler: StandardScaler,
    tmp_path: Path,
):
    """The functions create a `Porter` for estimators and transformers."""
    assert port(fitted_tree) == Estimator(fitted_tree).port()
    assert port(fitted_scaler) == Transformer(fitted_scaler).port()
    path = save(fitted_scaler, language='ruby', directory=tmp_path)
    assert path == str(tmp_path / 'StandardScaler.rb')
    # `test` isn't imported, otherwise pytest would collect it:
    assert {'make', 'test'} <= set(sklearn_porter.__all__)


def test_can_and_show():
    assert can(DecisionTreeClassifier, 'c', 'attached', 'predict_proba')
    assert not can(DecisionTreeClassifier, 'c', 'attached', 'transform')
    assert can(StandardScaler, 'c', 'attached', 'transform')
    assert not can(StandardScaler, 'c', 'attached', 'predict')
    assert not can(SplineTransformer)
    assert not can(Pipeline)
    table = show()
    assert 'DecisionTreeClassifier' in table
    assert 'StandardScaler' in table


def _fitted_models() -> dict:
    """All supported estimators and transformers and a pipeline, fitted."""
    from sklearn.datasets import load_diabetes
    from sklearn.pipeline import make_pipeline

    from tests.commons import CLASSIFIERS, OUTLIER_DETECTORS, REGRESSORS
    from tests.test_transformer import TRANSFORMERS

    x, y = load_iris(return_X_y=True)
    x_reg, y_reg = load_diabetes(return_X_y=True)
    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # e.g. of the convergence
        for candidate in CLASSIFIERS + REGRESSORS + OUTLIER_DETECTORS:
            est = candidate.create()
            # Few trees are enough for the generated code:
            for param in ('n_estimators', 'max_iter'):
                if param in est.get_params() and 'MLP' not in candidate.name:
                    est.set_params(**{param: 2})
            if candidate in CLASSIFIERS:
                est.fit(x, y)
            elif candidate in REGRESSORS:
                est.fit(x_reg[:, :4], y_reg)
            else:
                est.fit(x)
            models[candidate.name] = est
    for name, (create, _) in TRANSFORMERS.items():
        models[name] = create()
    models['pipeline'] = make_pipeline(StandardScaler(), SVC()).fit(x, y)
    return models


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
def test_no_trailing_whitespace(template: str):
    """The ported code has no lines which end with whitespace."""
    trailing = []
    for name, model in _fitted_models().items():
        for language in LANGUAGES:
            if not can(model, language, template):
                continue
            out = port(model, language=language, template=template)
            code = out[0] if isinstance(out, tuple) else out
            for line in code.splitlines():
                if line != line.rstrip():
                    trailing.append(f'{name} ({language}): {line!r}')
                    break
    assert not trailing


def test_converter_of_combined_leaves():
    """The values of the leaves of the `combined` trees use the converter."""
    leaf = re.compile(r'(?:classes|values)\[\d+\] = ([^;\s]+)')
    unconverted = []
    for name, model in _fitted_models().items():
        if not can(model, 'js', 'combined') or 'pipeline' in name:
            continue
        code = port(
            model,
            language='js',
            template='combined',
            converter=lambda value: f'{value:.4f}',
        )
        assert isinstance(code, str)
        values = leaf.findall(code)
        if not all(re.fullmatch(r'-?\d+\.\d{4}', v) for v in values):
            unconverted.append(name)
    assert not unconverted
