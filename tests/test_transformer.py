import json
from importlib import import_module
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pytest
from joblib import dump

# scikit-learn
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OrdinalEncoder,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.svm import SVC

# sklearn-porter
from sklearn_porter import Estimator, Transformer, can, port, save, show
from sklearn_porter import exceptions as exception
from sklearn_porter.cli.command.port import main as port_main
from sklearn_porter.cli.command.save import main as save_main
from tests.commons import LANGUAGES

ESTIMATOR_MODULE = import_module('sklearn_porter.Base')
TEMPLATES = ['attached', 'combined', 'exported']


def _iris_with_missing_values() -> np.ndarray:
    x = load_iris().data.copy()
    x[::7, 0] = np.nan
    return x


# The transformers and their expected model data:
TRANSFORMERS = {
    'binarizer': (
        lambda: Binarizer(threshold=1.5).fit(load_iris().data),
        lambda t: dict(threshold=[1.5] * 4),
    ),
    'max_abs_scaler': (
        lambda: MaxAbsScaler().fit(load_iris().data),
        lambda t: dict(scale=t.scale_.tolist()),
    ),
    'min_max_scaler': (
        lambda: MinMaxScaler(feature_range=(-1, 2)).fit(load_iris().data),
        lambda t: dict(scale=t.scale_.tolist(), min=t.min_.tolist()),
    ),
    'robust_scaler': (
        lambda: RobustScaler().fit(load_iris().data),
        lambda t: dict(center=t.center_.tolist(), scale=t.scale_.tolist()),
    ),
    'simple_imputer': (
        lambda: SimpleImputer().fit(_iris_with_missing_values()),
        lambda t: dict(statistics=t.statistics_.tolist()),
    ),
    'standard_scaler': (
        lambda: StandardScaler().fit(load_iris().data),
        lambda t: dict(mean=t.mean_.tolist(), scale=t.scale_.tolist()),
    ),
    'function_transformer': (
        lambda: FunctionTransformer(np.log1p).fit(load_iris().data),
        lambda t: dict(),
    ),
    'normalizer': (
        lambda: Normalizer(norm='max').fit(load_iris().data),
        lambda t: dict(),
    ),
    'k_bins_discretizer': (
        lambda: KBinsDiscretizer(n_bins=4, encode='ordinal').fit(
            load_iris().data
        ),
        lambda t: dict(
            edges=np.concatenate([e[1:-1] for e in t.bin_edges_]).tolist(),
            offsets=[0, 3, 6, 9, 12],
        ),
    ),
    'ordinal_encoder': (
        lambda: OrdinalEncoder().fit(np.round(_iris_with_missing_values())),
        lambda t: dict(
            categories=np.concatenate(
                [c[~np.isnan(c)] for c in t.categories_]
            ).tolist(),
            offsets=np.cumsum(
                [0] + [np.sum(~np.isnan(c)) for c in t.categories_]
            ).tolist(),
            missing=[1, 0, 0, 0],
        ),
    ),
}


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
    assert not can(SplineTransformer)


def test_unfitted_transformer():
    with pytest.raises(exception.NotFittedEstimatorError):
        Transformer(StandardScaler())


def test_unsupported_transformer():
    spline = SplineTransformer().fit(load_iris().data)
    with pytest.raises(
        exception.NotSupportedYetError, match='`SplineTransformer`'
    ):
        Transformer(spline)


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


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('name', TRANSFORMERS.keys())
def test_port_of_all_transformers(name: str, language: str, template: str):
    create, model_data = TRANSFORMERS[name]
    orig_est = create()
    assert can(orig_est, language, template, 'transform')
    assert not can(orig_est, language, template, 'predict')
    est = Transformer(orig_est, language=language, template=template)
    assert est.class_name == type(orig_est).__name__
    out = est.port()
    expected = model_data(orig_est)
    if template == 'exported':
        out, data = out
        assert json.loads(data) == expected
    elif template == 'attached':
        # The model data is in the source code:
        for values in expected.values():
            assert str(values[0]) in out
    assert 'Execute a transformation' in out
    method = 'Transform' if language in ('cs', 'go') else 'transform'
    assert f'{method}(' in out


@pytest.mark.parametrize(
    'params',
    [dict(with_centering=False), dict(with_scaling=False)],
    ids=['without_centering', 'without_scaling'],
)
def test_model_data_of_robust_scaler(params: dict):
    """Subtracting zero and dividing by one doesn't change the features."""
    orig_est = RobustScaler(**params).fit(load_iris().data)
    _, data = Transformer(orig_est, template='exported').port()
    data = json.loads(data)
    if orig_est.with_centering:
        assert data['center'] == orig_est.center_.tolist()
        assert data['scale'] == [1.0] * 4
    else:
        assert data['center'] == [0.0] * 4
        assert data['scale'] == orig_est.scale_.tolist()


@pytest.mark.parametrize('language', LANGUAGES)
def test_clip_of_min_max_scaler(language: str):
    """The bounds of `clip=True` are numbers in the code."""
    x = load_iris().data
    out = port(MinMaxScaler(feature_range=(-1, 2)).fit(x), language=language)
    assert '-1.0' not in out
    out = port(
        MinMaxScaler(feature_range=(-1, 2), clip=True).fit(x),
        language=language,
    )
    assert '-1.0' in out
    assert '2.0' in out


@pytest.mark.parametrize('language', LANGUAGES)
def test_missing_values_of_simple_imputer(language: str):
    """Missing values are `NaN` or a number in the code."""
    x = load_iris().data
    out = port(SimpleImputer(missing_values=-1).fit(x), language=language)
    assert '== -1.0' in out
    out = port(SimpleImputer().fit(x), language=language)
    assert '== -1.0' not in out


def test_simple_imputer_with_empty_features():
    """Dropped features (without observed values) aren't supported."""
    x = _iris_with_missing_values()
    x[:, 2] = np.nan
    orig_est = SimpleImputer().fit(x)
    with pytest.raises(exception.NotSupportedYetError, match='drops'):
        Transformer(orig_est)
    orig_est = SimpleImputer(keep_empty_features=True).fit(x)
    _, data = Transformer(orig_est, template='exported').port()
    assert json.loads(data)['statistics'][2] == 0.0


@pytest.mark.parametrize(
    'params, match',
    [
        (dict(add_indicator=True), '`add_indicator=True`'),
        (dict(missing_values=None, strategy='constant'), '`missing_values`'),
        (dict(strategy='constant', fill_value='x'), 'numbers as statistics'),
    ],
    ids=['add_indicator', 'missing_values_none', 'fill_value_string'],
)
def test_unsupported_simple_imputer(params: dict, match: str):
    if 'fill_value' in params or params.get('missing_values', 0) is None:
        x = np.array([['a', 'b'], [None, 'c']], dtype=object)
    else:
        x = _iris_with_missing_values()
    orig_est = SimpleImputer(**params).fit(x)
    with pytest.raises(exception.NotSupportedYetError, match=match):
        Transformer(orig_est)


def test_model_data_with_missing_values():
    """Model data with missing values (e.g. an empty feature) raises."""
    x = load_iris().data.copy()
    x[:, 1] = np.nan
    with pytest.raises(exception.NotSupportedYetError, match='`mean`'):
        Transformer(StandardScaler().fit(x))


def test_unfitted_binarizer():
    """A `Binarizer` passes `check_is_fitted`, but has no features."""
    with pytest.raises(exception.NotFittedEstimatorError):
        Transformer(Binarizer())


def test_show_of_all_transformers():
    table = show()
    for name in ('Binarizer', 'MaxAbsScaler', 'MinMaxScaler'):
        assert name in table
    for name in ('RobustScaler', 'SimpleImputer', 'StandardScaler'):
        assert name in table
    for name in ('FunctionTransformer', 'KBinsDiscretizer'):
        assert name in table
    for name in ('Normalizer', 'OrdinalEncoder'):
        assert name in table


@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'func', [None, np.abs, np.sqrt, np.square, np.exp, np.log, np.log1p]
)
def test_functions_of_function_transformer(func, language: str):
    """The function is chosen in the code of the template."""
    x = load_iris().data
    out = port(FunctionTransformer(func).fit(x), language=language)
    assert '{%' not in out
    if func is np.log1p and language in ('cs', 'ruby'):
        # Both languages have no function `log1p`:
        assert 'u - 1.0' in out
    elif func is np.square:
        x_i = '$features[$i]' if language == 'php' else 'features[i]'
        assert f'{x_i} * {x_i}' in out


@pytest.mark.parametrize(
    'params',
    [dict(func=np.sin), dict(func=np.log1p, kw_args=dict(out=None))],
    ids=['unsupported_function', 'kw_args'],
)
def test_unsupported_function_transformer(params: dict):
    orig_est = FunctionTransformer(**params).fit(load_iris().data)
    with pytest.raises(exception.NotSupportedYetError):
        Transformer(orig_est)


@pytest.mark.parametrize('norm', ['l1', 'l2'])
def test_unsupported_norm_of_normalizer(norm: str):
    orig_est = Normalizer(norm=norm).fit(load_iris().data)
    with pytest.raises(exception.NotSupportedYetError, match="'max'"):
        Transformer(orig_est)


@pytest.mark.parametrize('encode', ['onehot', 'onehot-dense'])
def test_unsupported_encoding_of_k_bins_discretizer(encode: str):
    orig_est = KBinsDiscretizer(n_bins=3, encode=encode).fit(load_iris().data)
    with pytest.raises(exception.NotSupportedYetError, match="'ordinal'"):
        Transformer(orig_est)


@pytest.mark.parametrize('language', LANGUAGES)
def test_unknown_and_missing_values_of_ordinal_encoder(language: str):
    """The values of unknown and missing categories are numbers or `NaN`."""
    x = np.round(_iris_with_missing_values())
    orig_est = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        encoded_missing_value=-2,
    ).fit(x)
    out = port(orig_est, language=language)
    assert '-1.0' in out
    assert '-2.0' in out
    nan = dict(
        c='NAN',
        cs='double.NaN',
        go='math.NaN()',
        java='Double.NaN',
        js='NaN',
        php='NAN',
        ruby='Float::NAN',
    )[language]
    assert nan in port(OrdinalEncoder().fit(x), language=language)


@pytest.mark.parametrize(
    'x, params, match',
    [
        (np.array([['a'], ['b']], dtype=object), dict(), 'numbers'),
        (np.array([[1], [1], [2]]), dict(max_categories=1), 'infrequent'),
    ],
    ids=['strings', 'infrequent_categories'],
)
def test_unsupported_ordinal_encoder(x: np.ndarray, params: dict, match: str):
    orig_est = OrdinalEncoder(**params).fit(x)
    with pytest.raises(exception.NotSupportedYetError, match=match):
        Transformer(orig_est)
