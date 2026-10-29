import json
from importlib import import_module
from pathlib import Path
from subprocess import CompletedProcess, run

import numpy as np
import pytest
from joblib import dump
from sklearn.compose import ColumnTransformer

# scikit-learn
from sklearn.datasets import load_iris
from sklearn.feature_selection import (
    RFE,
    RFECV,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import Estimator, Transformer, can, port, save, show
from sklearn_porter import exceptions as exception
from sklearn_porter.cli.command.port import main as port_main
from sklearn_porter.cli.command.save import main as save_main
from sklearn_porter.transformer.Normalizer import _get_sum_tree
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
        lambda t: dict(
            inputs=[0, 1, 2, 3],
            statistics=t.statistics_.tolist(),
            indicators=[0, 0, 0, 0],
        ),
    ),
    'simple_imputer_with_indicator': (
        lambda: SimpleImputer(add_indicator=True).fit(
            _iris_with_missing_values()
        ),
        lambda t: dict(
            inputs=[0, 1, 2, 3, 0],
            statistics=[*t.statistics_.tolist(), 0.0],
            indicators=[0, 0, 0, 0, 1],
        ),
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
    'one_hot_encoder': (
        lambda: OneHotEncoder().fit(np.round(_iris_with_missing_values())),
        lambda t: dict(
            inputs=[i for i, c in enumerate(t.categories_) for _ in c],
            categories=np.nan_to_num(np.concatenate(t.categories_)).tolist(),
            missing=np.isnan(np.concatenate(t.categories_))
            .astype(int)
            .tolist(),
        ),
    ),
    'polynomial_features': (
        lambda: PolynomialFeatures(2).fit(load_iris().data[:, :3]),
        lambda t: dict(
            # 1, x0, x1, x2, x0 x0, x1 x0, x2 x0, x1 x1, x2 x1, x2 x2 (a term
            # of the previous degree multiplied by a feature):
            factors=[0, 1, 2, 0, 0, 1, 0, 2, 0, 1, 1, 2, 1, 2, 2],
            offsets=[0, 0, 1, 2, 3, 5, 7, 9, 11, 13, 15],
        ),
    ),
    'target_encoder': (
        lambda: TargetEncoder(random_state=0).fit(
            np.round(_iris_with_missing_values()), load_iris().target % 2
        ),
        lambda t: dict(
            inputs=[0, 1, 2, 3],
            categories=np.concatenate(
                [c[~np.isnan(c)] for c in t.categories_]
            ).tolist(),
            encodings=np.concatenate(
                [
                    e[~np.isnan(c)]
                    for c, e in zip(t.categories_, t.encodings_, strict=True)
                ]
            ).tolist(),
            offsets=np.cumsum(
                [0] + [np.sum(~np.isnan(c)) for c in t.categories_]
            ).tolist(),
            missing=[1, 0, 0, 0],
            nans=[float(t.encodings_[0][-1]), 0.0, 0.0, 0.0],
            unknowns=[float(t.target_mean_)] * 4,
        ),
    ),
    'power_transformer': (
        lambda: PowerTransformer().fit(load_iris().data),
        lambda t: dict(
            lambdas=t.lambdas_.tolist(),
            mean=t._scaler.mean_.tolist(),
            scale=t._scaler.scale_.tolist(),
        ),
    ),
    'power_transformer_box_cox': (
        lambda: PowerTransformer(method='box-cox', standardize=False).fit(
            load_iris().data
        ),
        lambda t: dict(
            lambdas=t.lambdas_.tolist(), mean=[0.0] * 4, scale=[1.0] * 4
        ),
    ),
    'quantile_transformer': (
        lambda: QuantileTransformer(n_quantiles=20).fit(load_iris().data),
        lambda t: dict(
            quantiles=t.quantiles_.T.ravel().tolist(),
            references=t.references_.tolist(),
        ),
    ),
    'k_bins_discretizer_onehot': (
        lambda: KBinsDiscretizer(n_bins=4, encode='onehot-dense').fit(
            load_iris().data
        ),
        lambda t: dict(
            edges=np.concatenate([e[1:-1] for e in t.bin_edges_]).tolist(),
            offsets=[0, 3, 6, 9, 12],
        ),
    ),
}


# The feature selectors (of the iris dataset), their model data are the
# indices of the selected features:
_SELECTORS = dict(
    variance_threshold=lambda: VarianceThreshold(threshold=0.2),
    select_k_best=lambda: SelectKBest(k=2),
    select_percentile=lambda: SelectPercentile(percentile=50),
    generic_univariate_select=lambda: GenericUnivariateSelect(
        mode='k_best', param=3
    ),
    select_fpr=lambda: SelectFpr(alpha=1e-50),
    select_fdr=lambda: SelectFdr(alpha=1e-50),
    select_fwe=lambda: SelectFwe(alpha=1e-50),
    select_from_model=lambda: SelectFromModel(
        LogisticRegression(max_iter=1000)
    ),
    rfe=lambda: RFE(LogisticRegression(max_iter=1000), n_features_to_select=2),
    rfecv=lambda: RFECV(LogisticRegression(max_iter=1000), cv=3),
    sequential_feature_selector=lambda: SequentialFeatureSelector(
        LogisticRegression(max_iter=1000), n_features_to_select=3, cv=3
    ),
)
for _name, _create in _SELECTORS.items():
    TRANSFORMERS[_name] = (
        lambda create=_create: create().fit(*load_iris(return_X_y=True)),
        lambda t: dict(inputs=t.get_support(indices=True).tolist()),
    )


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


@pytest.mark.parametrize('add_indicator', [False, True])
def test_simple_imputer_with_empty_features(add_indicator: bool):
    """Features without observed values are dropped like in scikit-learn."""
    x = _iris_with_missing_values()
    x[:, 2] = np.nan
    orig_est = SimpleImputer(add_indicator=add_indicator).fit(x)
    est = Transformer(orig_est)
    expected = dict(
        inputs=[0, 1, 3],
        statistics=orig_est.statistics_[[0, 1, 3]].tolist(),
        indicators=[0, 0, 0],
    )
    if add_indicator:
        # The indicators of the features with missing values in the fit:
        expected = dict(
            inputs=[0, 1, 3, 0, 2],
            statistics=[*expected['statistics'], 0.0, 0.0],
            indicators=[0, 0, 0, 1, 1],
        )
    assert est._estimator.model_data == expected
    meta_info = est._estimator.meta_info
    assert meta_info['n_features_out'] == orig_est.transform(x[:1]).shape[1]
    orig_est = SimpleImputer(keep_empty_features=True).fit(x)
    _, data = Transformer(orig_est, template='exported').port()
    assert json.loads(data)['statistics'][2] == 0.0


def test_simple_imputer_without_features():
    x = np.full((3, 2), np.nan)
    orig_est = SimpleImputer().fit(x)
    with pytest.raises(exception.NotSupportedYetError, match='no transformed'):
        Transformer(orig_est)


@pytest.mark.parametrize(
    'params, match',
    [
        (dict(missing_values=None, strategy='constant'), '`missing_values`'),
        (dict(strategy='constant', fill_value='x'), 'numbers as statistics'),
    ],
    ids=['missing_values_none', 'fill_value_string'],
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
    for name in ('OneHotEncoder', 'PolynomialFeatures', 'TargetEncoder'):
        assert name in table
    for name in ('PowerTransformer',):
        assert name in table
    assert 'QuantileTransformer' in table
    for name in ('GenericUnivariateSelect', 'RFE', 'RFECV', 'SelectFdr'):
        assert name in table
    for name in ('SelectFpr', 'SelectFromModel', 'SelectFwe', 'SelectKBest'):
        assert name in table
    for name in (
        'SelectPercentile',
        'SequentialFeatureSelector',
        'VarianceThreshold',
    ):
        assert name in table


@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'func',
    [
        None,
        np.abs,
        np.sqrt,
        np.square,
        np.exp,
        np.log,
        np.log1p,
        np.negative,
        np.reciprocal,
        np.floor,
        np.ceil,
        np.sign,
    ],
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


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'kw_args, bounds',
    [
        (dict(a_min=-1, a_max=2), ('-1.0', '2.0')),
        (dict(min=0), ('0.0', None)),
        (dict(a_min=None, a_max=1.5), (None, '1.5')),
    ],
    ids=['both', 'lower', 'upper'],
)
def test_clip_of_function_transformer(
    kw_args: dict, bounds: tuple, language: str, template: str
):
    """The bounds of `np.clip` are numbers in the code."""
    orig_est = FunctionTransformer(np.clip, kw_args=kw_args)
    orig_est.fit(load_iris().data)
    est = Transformer(orig_est, language=language, template=template)
    out = est.port()
    out = out[0] if isinstance(out, tuple) else out
    assert '{%' not in out
    for bound in bounds:
        if bound is not None:
            assert bound in out


@pytest.mark.parametrize(
    'params',
    [
        dict(func=np.sin),
        dict(func=np.log1p, kw_args=dict(out=None)),
        dict(func=np.clip, kw_args=dict(a_min=0)),
        dict(func=np.clip, kw_args=dict(min=None, max=None)),
        dict(func=np.clip, kw_args=dict(min=2, max=1)),
        dict(func=np.clip, kw_args=dict(min=[0, 1, 2, 3])),
        dict(func=np.clip, kw_args=dict(min=np.nan)),
    ],
    ids=[
        'unsupported_function',
        'kw_args',
        'clip_without_a_max',
        'clip_without_bounds',
        'clip_with_invalid_bounds',
        'clip_with_arrays',
        'clip_with_nan',
    ],
)
def test_unsupported_function_transformer(params: dict):
    orig_est = FunctionTransformer(**params).fit(load_iris().data)
    with pytest.raises(exception.NotSupportedYetError):
        Transformer(orig_est)


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('norm', ['l1', 'l2'])
def test_norms_of_normalizer(norm: str, language: str, template: str):
    """The norms `l1` and `l2` sum the features in the order of NumPy."""
    orig_est = Normalizer(norm=norm).fit(load_iris().data)
    out = port(orig_est, language=language, template=template)
    out = out[0] if isinstance(out, tuple) else out
    assert '{%' not in out
    x = '$features' if language == 'php' else 'features'
    if norm == 'l2':
        term = f'{x}[0]*{x}[0]' if language == 'go' else f'{x}[0] * {x}[0]'
        assert term in out
    else:
        assert f'{x}[3]' in out


@pytest.mark.parametrize(
    'n, expected',
    [
        (1, 0),
        (3, ((0, 1), 2)),
        (9, ((((0, 1), (2, 3)), ((4, 5), (6, 7))), 8)),
        (
            16,
            (
                (((0, 8), (1, 9)), ((2, 10), (3, 11))),
                (((4, 12), (5, 13)), ((6, 14), (7, 15))),
            ),
        ),
    ],
)
def test_sum_tree_of_normalizer(n: int, expected):
    """The order of the pairwise summation of NumPy."""
    assert _get_sum_tree(0, n) == expected


def test_sum_tree_of_normalizer_with_many_features():
    """More than 128 features are summed in two halves."""

    def leaves(tree) -> list:
        if isinstance(tree, int):
            return [tree]
        return leaves(tree[0]) + leaves(tree[1])

    tree = _get_sum_tree(0, 300)
    assert isinstance(tree, tuple)
    assert sorted(leaves(tree)) == list(range(300))
    assert sorted(leaves(tree[0])) == list(range(144))


@pytest.mark.parametrize(
    'create, n_features_out',
    [
        (lambda: OneHotEncoder(), lambda t: len(t.get_feature_names_out())),
        (
            lambda: OneHotEncoder(drop='first', handle_unknown='ignore'),
            lambda t: len(t.get_feature_names_out()),
        ),
        (lambda: PolynomialFeatures(3), lambda t: t.n_output_features_),
        (
            lambda: KBinsDiscretizer(n_bins=3, encode='onehot'),
            lambda t: int(sum(t.n_bins_)),
        ),
        (
            lambda: KBinsDiscretizer(n_bins=3, encode='ordinal'),
            lambda t: t.n_features_in_,
        ),
    ],
    ids=[
        'one_hot_encoder',
        'one_hot_encoder_with_drop',
        'polynomial_features',
        'k_bins_discretizer_onehot',
        'k_bins_discretizer_ordinal',
    ],
)
def test_number_of_transformed_features(create, n_features_out):
    """Some transformers change the number of features."""
    orig_est = create().fit(np.round(load_iris().data))
    meta_info = Transformer(orig_est)._estimator.meta_info
    assert meta_info['n_features'] == 4
    assert meta_info['n_features_out'] == n_features_out(orig_est)


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('encode', ['onehot', 'onehot-dense'])
def test_onehot_encoding_of_k_bins_discretizer(
    encode: str, language: str, template: str
):
    orig_est = KBinsDiscretizer(n_bins=3, encode=encode).fit(load_iris().data)
    out = port(orig_est, language=language, template=template)
    out = out[0] if isinstance(out, tuple) else out
    assert '{%' not in out
    if template == 'combined':
        # The bins of the first feature (the first bin is unbounded below):
        x = '$features[0]' if language == 'php' else 'features[0]'
        edges = orig_est.bin_edges_[0]
        assert f'{x} < {float(edges[1])!r}' in out
        assert f'{x} >= {float(edges[1])!r}' in out


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_drop_of_one_hot_encoder(language: str, template: str):
    """The dropped categories are no transformed features."""
    x = np.array([[0, 1], [1, 2], [2, 1]], dtype=float)
    orig_est = OneHotEncoder(drop=[0, 2]).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    assert est._estimator.model_data == dict(
        inputs=[0, 0, 1], categories=[1.0, 2.0, 1.0], missing=[0, 0, 0]
    )
    out = est.port()
    out = out[0] if isinstance(out, tuple) else out
    assert '{%' not in out


@pytest.mark.parametrize(
    'x, params, match',
    [
        (np.array([['a'], ['b']], dtype=object), dict(), 'numbers'),
        (np.array([[1], [1], [2]]), dict(max_categories=1), 'infrequent'),
        (np.array([[1], [1]]), dict(drop='first'), 'no transformed'),
    ],
    ids=['strings', 'infrequent_categories', 'no_features'],
)
def test_unsupported_one_hot_encoder(x: np.ndarray, params: dict, match: str):
    orig_est = OneHotEncoder(**params).fit(x)
    with pytest.raises(exception.NotSupportedYetError, match=match):
        Transformer(orig_est)


@pytest.mark.parametrize(
    'params',
    [
        dict(degree=1),
        dict(degree=3),
        dict(degree=3, include_bias=False),
        dict(degree=3, interaction_only=True),
        dict(degree=(2, 3)),
        dict(degree=(2, 3), include_bias=False),
        dict(degree=(3, 3), interaction_only=True),
    ],
)
def test_factors_of_polynomial_features(params: dict):
    """The factors of each transformed feature match the powers."""
    orig_est = PolynomialFeatures(**params).fit(load_iris().data)
    model_data = Transformer(orig_est)._estimator.model_data
    offsets, factors = model_data['offsets'], model_data['factors']
    powers = np.zeros_like(orig_est.powers_)
    for i in range(len(offsets) - 1):
        for j in factors[offsets[i] : offsets[i + 1]]:
            powers[i, j] += 1
    assert np.array_equal(powers, orig_est.powers_)


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_combined_template_of_polynomial_features(language: str, template: str):
    """The products are multiplied in the order of scikit-learn."""
    orig_est = PolynomialFeatures(3).fit(load_iris().data[:, :2])
    out = port(orig_est, language=language, template=template)
    out = out[0] if isinstance(out, tuple) else out
    assert '{%' not in out
    if template == 'combined':
        x = '$features' if language == 'php' else 'features'
        # x0² x1 is (x1 x0) x0:
        assert f'{x}[1] * {x}[0] * {x}[0]' in out


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_multiclass_target_encoder(language: str, template: str):
    """One transformed feature per feature and class."""
    x = np.round(load_iris().data)
    orig_est = TargetEncoder(random_state=0).fit(x, load_iris().target)
    est = Transformer(orig_est, language=language, template=template)
    model_data = est._estimator.model_data
    assert model_data['inputs'] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert model_data['unknowns'] == orig_est.target_mean_.tolist() * 4
    assert est._estimator.meta_info['n_features_out'] == 12
    out = est.port()
    out = out[0] if isinstance(out, tuple) else out
    assert '{%' not in out


def test_unsupported_target_encoder():
    x = np.array([['a'], ['b'], ['a'], ['b']], dtype=object)
    orig_est = TargetEncoder().fit(x, [0, 1, 0, 1])
    with pytest.raises(exception.NotSupportedYetError, match='numbers'):
        Transformer(orig_est)


@pytest.mark.parametrize('language', LANGUAGES)
def test_formulas_of_power_transformer(language: str):
    """The template `combined` contains only the formulas of the lambdas."""
    x = load_iris().data
    orig_est = PowerTransformer(standardize=False).fit(x)
    orig_est.lambdas_ = np.array([0.0, 2.0, 0.5, -1.5])
    out = port(orig_est, language=language, template='combined')
    assert '{%' not in out
    assert '3.5' in out  # 2 - lambda of the last feature
    orig_est = PowerTransformer(method='box-cox', standardize=False).fit(x)
    out = port(orig_est, language=language, template='combined')
    assert '709.78' in out


@pytest.mark.parametrize('language', LANGUAGES)
def test_combined_template_of_quantile_transformer(language: str):
    """The quantiles of each feature are numbers in the code."""
    orig_est = QuantileTransformer(n_quantiles=5).fit(load_iris().data)
    out = port(orig_est, language=language, template='combined')
    assert '{%' not in out
    assert str(float(orig_est.quantiles_[1, 0])) in out


def test_unsupported_quantile_transformer():
    orig_est = QuantileTransformer(n_quantiles=5, output_distribution='normal')
    orig_est.fit(load_iris().data)
    with pytest.raises(exception.NotSupportedYetError, match="'uniform'"):
        Transformer(orig_est)


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_selected_features_of_feature_selector(language: str, template: str):
    """Each transformed feature is a selected feature."""
    x, y = load_iris(return_X_y=True)
    orig_est = SelectKBest(k=2).fit(x, y)
    est = Transformer(orig_est, language=language, template=template)
    assert est._estimator.model_data == dict(inputs=[2, 3])
    assert est._estimator.meta_info == dict(n_features=4, n_features_out=2)
    out = est.port()
    out = out[0] if isinstance(out, tuple) else out
    assert '{%' not in out
    if template == 'combined':
        x = '$features' if language == 'php' else 'features'
        assert f'{x}[3]' in out
        assert f'{x}[0]' not in out


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'create, allow_nan',
    [
        (lambda: SelectKBest(k=2), False),
        (lambda: VarianceThreshold(), True),
        (lambda: RFE(LogisticRegression(max_iter=1000)), False),
        (lambda: RFE(DecisionTreeClassifier()), True),
    ],
    ids=['k_best', 'variance', 'rfe_linear', 'rfe_tree'],
)
def test_invalid_values_of_feature_selector(
    create, allow_nan: bool, language: str, template: str
):
    """The check of the values like scikit-learn (tag `allow_nan`)."""
    x, y = load_iris(return_X_y=True)
    est = Transformer(create().fit(x, y), language=language, template=template)
    assert est._estimator.allow_nan == allow_nan
    out = est.port()
    out = out[0] if isinstance(out, tuple) else out
    assert ('NaN or infinity' in out) != allow_nan


def test_feature_selector_without_selected_features():
    x, y = load_iris(return_X_y=True)
    orig_est = SelectFwe(alpha=1e-300).fit(x, y)
    assert not orig_est.get_support().any()
    with pytest.raises(exception.NotSupportedYetError, match='no feature'):
        Transformer(orig_est)


@pytest.mark.parametrize(
    'orig_est',
    [
        SelectKBest(k=2),
        VarianceThreshold(),
        RFE(LogisticRegression()),
        SelectFromModel(LogisticRegression()),
        SelectFromModel(LogisticRegression(), prefit=True),
    ],
    ids=['k_best', 'variance', 'rfe', 'from_model', 'from_model_prefit'],
)
def test_unfitted_feature_selector(orig_est):
    with pytest.raises(exception.NotFittedEstimatorError):
        Transformer(orig_est)


def test_select_from_model_with_prefit_estimator():
    """scikit-learn selects the features without a fit of the selector."""
    x, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=1000).fit(x, y)
    orig_est = SelectFromModel(clf, prefit=True)
    expected = orig_est.get_support(indices=True).tolist()
    est = Transformer(orig_est)
    assert est._estimator.model_data == dict(inputs=expected)
    assert est._estimator.meta_info['n_features'] == 4


def test_polynomial_features_with_bias_only():
    """The bias is the only transformed feature (with a dummy factor)."""
    orig_est = PolynomialFeatures(0).fit(load_iris().data)
    est = Transformer(orig_est)
    assert est._estimator.model_data == dict(factors=[0], offsets=[0, 0])
    assert est._estimator.meta_info['n_features_out'] == 1


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


# The spellings of infinite and missing values in the samples, e.g. of
# Python (`str`) and of the ported transformers:
SPELLINGS = [
    'inf',
    '+inf',
    '-inf',
    'INF',
    'infinity',
    'Infinity',
    '-Infinity',
    'nan',
    'NaN',
]


@pytest.mark.regression
@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_infinite_values_in_samples(
    tmp_root_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    template: str,
):
    """The ported transformers read all spellings of infinite values."""
    cmds = []

    def run_and_record(cmd: list[str], **kwargs) -> CompletedProcess:
        cmds.append(cmd)
        return run(cmd, **kwargs)

    monkeypatch.setattr(ESTIMATOR_MODULE, 'run', run_and_record)

    x = load_iris().data
    orig_est = FunctionTransformer(np.negative).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    tmp_dir = tmp_root_dir / f'infinite_values_in_samples-{language}-{template}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    est.make(x[:1], directory=tmp_dir, delete_created_files=False)
    samples = [[spelling, '1.5', '-0.0', spelling] for spelling in SPELLINGS]
    x_true = orig_est.transform([[float(v) for v in s] for s in samples])
    # Several samples from the standard input:
    proc = run(
        cmds[0],
        input='\n'.join(' '.join(s) for s in samples) + '\n',
        capture_output=True,
        encoding='utf-8',
        check=True,
    )
    lines = proc.stdout.strip().split('\n')
    x_tr = np.array([[float(v) for v in line.split()] for line in lines])
    np.testing.assert_array_equal(x_tr, x_true)
    # One sample from the arguments:
    for sample, values in zip(samples, x_true, strict=True):
        proc = run(
            cmds[0] + sample, capture_output=True, encoding='utf-8', check=True
        )
        values_tr = [float(v) for v in proc.stdout.split()]
        np.testing.assert_array_equal(values_tr, values)


def _composite_transformers() -> dict:
    """Fitted composite transformers (with pipelines, weights and drops)."""
    x = load_iris().data
    return dict(
        column_transformer=ColumnTransformer(
            [
                ('scaler', StandardScaler(), [0, 1]),
                ('drop', 'drop', [2]),
                ('pipe', make_pipeline(MinMaxScaler(), Binarizer()), [3]),
            ],
            remainder='passthrough',
            transformer_weights={'scaler': 2.0},
        ).fit(x),
        feature_union=FeatureUnion(
            [
                ('scaler', StandardScaler()),
                ('passthrough', 'passthrough'),
                ('drop', 'drop'),
                ('poly', PolynomialFeatures(2)),
            ]
        ).fit(x),
    )


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('name', ['column_transformer', 'feature_union'])
def test_port_of_composite_transformers(
    name: str, language: str, template: str
):
    """The transformers of the parts are inserted into one source file."""
    orig_est = _composite_transformers()[name]
    assert can(orig_est, language, template, 'transform')
    est = Transformer(orig_est, language=language, template=template)
    n_features_out = orig_est.transform(load_iris().data[:1]).shape[1]
    assert est._estimator.meta_info['n_features_out'] == n_features_out
    out = est.port()
    if template == 'exported':
        out, data = out
        # The transformers of all parts (without the dropped part and the
        # empty remainder of the column transformer):
        assert len(json.loads(data)['transformers']) == 3
    assert '{%' not in out
    assert '{{ ' not in out
    if language == 'c':
        assert 'void t1_1_transform(' in out
    else:
        assert type(orig_est).__name__ + 'StandardScaler' in out


def test_unsupported_part_of_column_transformer():
    x = load_iris().data
    orig_est = ColumnTransformer(
        [
            ('scaler', StandardScaler(), [0]),
            ('spline', SplineTransformer(), [1]),
        ]
    ).fit(x)
    assert not Transformer.supports(orig_est)
    with pytest.raises(exception.NotSupportedYetError, match='spline'):
        Transformer(orig_est)
    orig_est = FeatureUnion([('spline', SplineTransformer())]).fit(x)
    assert not Transformer.supports(orig_est)
