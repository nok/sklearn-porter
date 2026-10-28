from pathlib import Path

import pytest
from joblib import dump

# scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import Estimator, Pipeline, Transformer, can, port, save
from sklearn_porter import exceptions as exception
from sklearn_porter.cli.command.port import main as port_main
from sklearn_porter.cli.command.save import main as save_main


def _fit(steps: list) -> SklearnPipeline:
    x, y = load_iris(return_X_y=True)
    return SklearnPipeline(steps).fit(x, y)


@pytest.fixture(scope='module')
def fitted_pipeline() -> SklearnPipeline:
    return _fit(
        [
            ('scaler', StandardScaler()),
            ('skipped', None),
            ('scaler2', StandardScaler(with_mean=False)),
            ('tree', DecisionTreeClassifier(random_state=0)),
        ]
    )


def test_pipeline(fitted_pipeline: SklearnPipeline):
    est = Pipeline(fitted_pipeline)
    assert est.estimator is fitted_pipeline
    # The skipped step isn't ported:
    assert [type(step) for step in est.steps] == [
        Transformer,
        Transformer,
        Estimator,
    ]
    assert est.steps[0].estimator is fitted_pipeline[0]
    assert est.steps[1].estimator is fitted_pipeline[2]
    assert est.steps[2].estimator is fitted_pipeline[-1]
    # The defaults of the final estimator:
    assert est.language == 'java'
    assert est.template == 'attached'
    assert est.class_name == 'Pipeline'
    assert {step.language for step in est.steps} == {'java'}
    assert 'pipeline\n--------\nname: Pipeline(StandardScaler, ' in repr(est)


def test_settings_of_all_steps(fitted_pipeline: SklearnPipeline):
    est = Pipeline(fitted_pipeline, language='go', template='exported')
    assert {step.language for step in est.steps} == {'go'}
    assert {step.template for step in est.steps} == {'exported'}
    est.language = 'php'
    est.template = 'combined'
    est.converter = repr
    assert {step.language for step in est.steps} == {'php'}
    assert {step.template for step in est.steps} == {'combined'}
    assert {step.converter for step in est.steps} == {repr}


def test_port(fitted_pipeline: SklearnPipeline):
    """The pipeline ports each step like `Transformer` and `Estimator`."""
    outs = Pipeline(fitted_pipeline, language='js').port()
    assert outs == [
        Transformer(fitted_pipeline[0], language='js').port(),
        Transformer(fitted_pipeline[2], language='js').port(
            class_name='StandardScaler2'
        ),
        Estimator(fitted_pipeline[-1], language='js').port(),
    ]
    outs = Pipeline(fitted_pipeline, template='exported').port(to_json=True)
    assert all(isinstance(out, tuple) and len(out) == 2 for out in outs)


def test_save(fitted_pipeline: SklearnPipeline, tmp_path: Path):
    """Repeated class names get a number, so the files are unique."""
    est = Pipeline(fitted_pipeline, language='c')
    assert [step.class_name for step in est.steps] == [
        'StandardScaler',
        'StandardScaler2',
        'DecisionTreeClassifier',
    ]
    paths = est.save(tmp_path)
    assert paths == [
        str(tmp_path / 'StandardScaler.c'),
        str(tmp_path / 'StandardScaler2.c'),
        str(tmp_path / 'DecisionTreeClassifier.c'),
    ]


def test_can(fitted_pipeline: SklearnPipeline):
    est = Pipeline(fitted_pipeline)
    assert est.can('c', 'attached', 'predict')
    assert est.can('c', 'attached', 'predict_proba')
    assert can(fitted_pipeline, 'ruby', 'exported', 'predict_proba')
    assert not can(SklearnPipeline)  # the steps are unknown
    svc = _fit([('scaler', StandardScaler()), ('svm', SVC())])
    # The SVMs don't support the template `combined`:
    assert not Pipeline.supports(svc, 'c', 'combined')
    with pytest.raises(exception.NotSupportedYetError, match='all steps'):
        Pipeline(svc, template='combined')


def test_pipeline_of_search():
    x, y = load_iris(return_X_y=True)
    pipeline = SklearnPipeline([('scaler', StandardScaler()), ('svm', SVC())])
    search = GridSearchCV(pipeline, {'svm__C': [0.1, 1.0]}, cv=2).fit(x, y)
    est = Pipeline(search)
    assert est.estimator is search.best_estimator_
    with pytest.raises(ValueError, match='not a pipeline'):
        Pipeline(GridSearchCV(pipeline, {'svm__C': [0.1, 1.0]}, cv=2))


@pytest.mark.parametrize('obj', [None, object, 0, 'string', SVC()])
def test_invalid_pipeline(obj):
    with pytest.raises(ValueError, match='not a pipeline'):
        Pipeline(obj)


def test_pipeline_without_final_estimator():
    pipeline = _fit([('scaler', StandardScaler()), ('final', 'passthrough')])
    with pytest.raises(ValueError, match='no final estimator'):
        Pipeline(pipeline)


def test_pipeline_without_preprocessing():
    pipeline = _fit([('skipped', 'passthrough'), ('svm', SVC())])
    with pytest.raises(ValueError, match='no preprocessing steps'):
        Pipeline(pipeline)
    assert not Pipeline.supports(pipeline)


def test_unsupported_steps():
    pipeline = _fit([('binarizer', Binarizer()), ('svm', SVC())])
    with pytest.raises(exception.NotSupportedYetError, match='`binarizer`'):
        Pipeline(pipeline)
    assert not Pipeline.supports(pipeline)


def test_unsupported_final_estimator():
    pipeline = _fit([('scaler', StandardScaler()), ('binarizer', Binarizer())])
    with pytest.raises(ValueError):
        Pipeline(pipeline)
    assert not Pipeline.supports(pipeline)


def test_unfitted_pipeline():
    pipeline = SklearnPipeline([('scaler', StandardScaler()), ('svm', SVC())])
    with pytest.raises(exception.NotFittedEstimatorError):
        Pipeline(pipeline)


def test_functions(fitted_pipeline: SklearnPipeline, tmp_path: Path):
    """The functions create a `Pipeline` (by `Porter`) for pipelines."""
    assert port(fitted_pipeline) == Pipeline(fitted_pipeline).port()
    paths = save(fitted_pipeline, language='ruby', directory=tmp_path)
    assert paths[-1] == str(tmp_path / 'DecisionTreeClassifier.rb')


def test_cli(fitted_pipeline: SklearnPipeline, tmp_path: Path):
    path = tmp_path / 'pipeline.joblib'
    dump(fitted_pipeline, str(path))
    out = port_main(
        dict(model=str(path), language='c', skip_warnings=True), silent=True
    )
    assert out.count('void transform(') == 2
    assert 'int predict_index(' in out
    out = save_main(
        dict(
            model=str(path),
            template='exported',
            directory=str(tmp_path),
            skip_warnings=True,
        ),
        silent=True,
    )
    assert str(tmp_path / 'StandardScaler.json') in out
    assert str(tmp_path / 'DecisionTreeClassifier.java') in out
