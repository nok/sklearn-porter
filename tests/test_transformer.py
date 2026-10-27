import json
from importlib import import_module
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pytest
from joblib import dump

# scikit-learn
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.svm import SVC

# sklearn-porter
from sklearn_porter import Estimator, Transformer, can, port, save, show
from sklearn_porter import exceptions as exception
from sklearn_porter.cli.command.port import main as port_main
from sklearn_porter.cli.command.save import main as save_main
from tests.commons import LANGUAGES

ESTIMATOR_MODULE = import_module('sklearn_porter.Base')
TEMPLATES = ['attached', 'combined', 'exported']


@pytest.fixture(scope='module')
def fitted_scaler() -> StandardScaler:
    return StandardScaler().fit(load_iris().data)


def test_transformer(fitted_scaler: StandardScaler):
    est = Transformer(fitted_scaler)
    assert est.transformer is est.estimator is fitted_scaler
    assert est.language == 'js'
    assert est.template == 'attached'
    assert est.class_name == 'StandardScaler'
    assert 'transformer\n-----------\nname: StandardScaler' in repr(est)
    # `transform()` is an alias of `make()`, the predictions of `Estimator`
    # aren't available:
    assert Transformer.transform is Transformer.make
    assert not hasattr(Transformer, 'predict')


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_port(fitted_scaler: StandardScaler, language: str, template: str):
    assert can(fitted_scaler, language, template)
    assert can(fitted_scaler, language, template, 'transform')
    assert not can(fitted_scaler, language, template, 'predict')
    est = Transformer(fitted_scaler, language=language, template=template)
    out = est.port()
    if template == 'exported':
        out, data = out
        assert json.loads(data) == dict(
            mean=np.asarray(fitted_scaler.mean_).tolist(),
            scale=np.asarray(fitted_scaler.scale_).tolist(),
        )
    else:
        # The model data is in the source code:
        assert str(np.asarray(fitted_scaler.mean_).tolist()[0]) in out
        assert str(np.asarray(fitted_scaler.scale_).tolist()[0]) in out
    assert 'Transformer:' in out
    assert 'Execute a transformation' in out
    assert 'Execute the transformations of several samples' in out
    method = 'Transform' if language in ('cs', 'go') else 'transform'
    assert f'{method}(' in out


@pytest.mark.parametrize('language', LANGUAGES)
def test_indentation_of_combined_template(
    fitted_scaler: StandardScaler, language: str
):
    """Test that the code of the `combined` template is indented with tabs."""
    est = Transformer(fitted_scaler, language=language, template='combined')
    ported = est.port()
    assert '\n\t' in ported
    assert not [line for line in ported.split('\n') if line.startswith('    ')]


@pytest.mark.parametrize(
    'params',
    [
        dict(with_mean=False),
        dict(with_std=False),
        dict(with_mean=False, with_std=False),
    ],
    ids=['without_mean', 'without_std', 'without_mean_and_std'],
)
def test_model_data_without_mean_or_std(params: dict):
    """Subtracting zero and dividing by one doesn't change the features."""
    x = load_iris().data
    orig_est = StandardScaler(**params).fit(x)
    _, data = Transformer(orig_est, template='exported').port()
    data = json.loads(data)
    n_features = x.shape[1]
    with_mean, with_std = orig_est.with_mean, orig_est.with_std
    mean = orig_est.mean_.tolist() if with_mean else [0.0] * n_features
    scale = orig_est.scale_.tolist() if with_std else [1.0] * n_features
    assert data == dict(mean=mean, scale=scale)


def test_module_functions(fitted_scaler: StandardScaler, tmp_path: Path):
    """The functions of the module create a `Transformer` for transformers."""
    assert 'this.transform = function' in port(fitted_scaler, language='js')
    paths = save(
        fitted_scaler, language='c', template='exported', directory=tmp_path
    )
    assert paths == (
        str(tmp_path / 'StandardScaler.c'),
        str(tmp_path / 'StandardScaler.json'),
    )


def test_show():
    assert 'StandardScaler' in show()
    assert 'StandardScaler' in show(language='c')


def test_can_with_classes():
    assert can(StandardScaler)
    assert can(StandardScaler, 'c', 'combined', 'transform')
    assert not can(StandardScaler, 'c', 'combined', 'predict')
    assert not can(Binarizer)


def test_unfitted_transformer():
    with pytest.raises(exception.NotFittedEstimatorError):
        Transformer(StandardScaler())


def test_unsupported_transformer():
    binarizer = Binarizer().fit(load_iris().data)
    with pytest.raises(exception.NotSupportedYetError, match='`Binarizer`'):
        Transformer(binarizer)


@pytest.mark.parametrize('obj', [None, object, 0, 'string'])
def test_invalid_transformer(obj):
    with pytest.raises(ValueError):
        Transformer(obj)


def test_estimator_as_transformer():
    x, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match='with `Estimator`'):
        Transformer(SVC().fit(x, y))


def test_transformer_as_estimator(fitted_scaler: StandardScaler):
    with pytest.raises(ValueError, match='with `Transformer`'):
        Estimator(fitted_scaler)


def test_pipeline_as_transformer():
    x, y = load_iris(return_X_y=True)
    pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC())]).fit(
        x, y
    )
    with pytest.raises(exception.NotSupportedYetError, match='pipeline'):
        Transformer(pipeline)


def test_make_reads_values(
    fitted_scaler: StandardScaler,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test the parsing of the values (one sample per line) of `make()`."""
    inputs = []

    def run(cmd: list[str], **kwargs) -> CompletedProcess:
        inputs.append(kwargs['input'])
        n_samples = len(kwargs['input'].splitlines())
        stdout = 'NaN -0 1.5 -Infinity\n' * n_samples
        return CompletedProcess(cmd, 0, stdout=stdout, stderr='')

    monkeypatch.setattr(ESTIMATOR_MODULE, 'run', run)
    est = Transformer(fitted_scaler)
    x = load_iris().data[:3]

    x_tr = est.make(x, check_dependencies=False, directory=tmp_path)
    assert inputs == [''.join(' '.join(map(str, e)) + '\n' for e in x)]
    expected = np.tile([np.nan, -0.0, 1.5, -np.inf], (3, 1))
    np.testing.assert_array_equal(x_tr, expected)
    assert np.signbit(x_tr[:, 1]).all()

    # A single sample:
    x_tr = est.transform(x[0], check_dependencies=False, directory=tmp_path)
    assert x_tr.shape == (4,)


def test_make_with_invalid_values(
    fitted_scaler: StandardScaler,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def run(cmd: list[str], **kwargs) -> CompletedProcess:
        return CompletedProcess(cmd, 0, stdout='1.0 x 2.0 3.0\n', stderr='')

    monkeypatch.setattr(ESTIMATOR_MODULE, 'run', run)
    monkeypatch.setattr(ESTIMATOR_MODULE, 'sleep', lambda _: None)
    est = Transformer(fitted_scaler)
    with pytest.raises(exception.ExecutionFailed, match='Invalid output'):
        est.make(
            load_iris().data[:1], check_dependencies=False, directory=tmp_path
        )


def test_cli(fitted_scaler: StandardScaler, tmp_path: Path):
    path = tmp_path / 'scaler.joblib'
    dump(fitted_scaler, str(path))
    out = port_main(
        dict(model=str(path), language='c', skip_warnings=True), silent=True
    )
    assert 'void transform(' in out
    out = save_main(
        dict(model=str(path), directory=str(tmp_path), skip_warnings=True),
        silent=True,
    )
    assert str(tmp_path / 'StandardScaler.js') in out
