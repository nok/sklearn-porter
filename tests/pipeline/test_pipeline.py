import json
import warnings
from importlib import import_module
from pathlib import Path
from subprocess import CompletedProcess, run

import numpy as np
import pytest

# scikit-learn
from sklearn.datasets import load_diabetes, load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline as SklearnPipeline
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
    StandardScaler,
    TargetEncoder,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import Pipeline, Porter
from sklearn_porter import exceptions as exception
from tests.commons import (
    CANDIDATES,
    LANGUAGES,
    OUTLIER_DETECTORS,
    REGRESSORS,
    Candidate,
)
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

BASE_MODULE = import_module('sklearn_porter.Base')
TEMPLATES = ['attached', 'combined', 'exported']

# The preprocessing steps (also two transformers of the same class):
PREPROCESSING = {
    'scaler': lambda: [('scaler', StandardScaler())],
    'two_scalers': lambda: [
        ('scaler', StandardScaler()),
        ('skipped', 'passthrough'),
        ('scaler2', StandardScaler(with_mean=False)),
    ],
    # All transformers (the imputer replaces the missing values of the
    # samples):
    'all_transformers': lambda: [
        ('imputer', SimpleImputer()),
        ('min_max', MinMaxScaler(feature_range=(-1, 2), clip=True)),
        ('max_abs', MaxAbsScaler()),
        ('robust', RobustScaler()),
        ('standard', StandardScaler()),
    ],
    'binarizer': lambda: [('binarizer', Binarizer(threshold=3.0))],
    'discretizers': lambda: [
        ('function', FunctionTransformer(np.sqrt)),
        ('k_bins', KBinsDiscretizer(n_bins=5, encode='ordinal')),
        ('ordinal', OrdinalEncoder()),
        ('normalizer', Normalizer(norm='max')),
    ],
    # Transformers which change the number of features:
    'one_hot_encoder': lambda: [
        ('k_bins', KBinsDiscretizer(n_bins=4, encode='ordinal')),
        ('one_hot', OneHotEncoder(drop='first')),
    ],
    'polynomial_features': lambda: [
        ('poly', PolynomialFeatures(2)),
        ('scaler', StandardScaler()),
    ],
    'k_bins_onehot': lambda: [
        ('k_bins', KBinsDiscretizer(n_bins=3, encode='onehot')),
    ],
    'target_encoder': lambda: [
        ('k_bins', KBinsDiscretizer(n_bins=4, encode='ordinal')),
        ('target', TargetEncoder(random_state=0)),
    ],
    'power_transformer': lambda: [('power', PowerTransformer())],
    'quantile_transformer': lambda: [
        ('quantile', QuantileTransformer(n_quantiles=50)),
    ],
    # The indicators of the missing values (of the samples of the fit):
    'imputer_with_indicator': lambda: [
        ('imputer', SimpleImputer(add_indicator=True)),
    ],
    'normalizers': lambda: [
        ('l1', Normalizer(norm='l1')),
        (
            'function',
            FunctionTransformer(np.clip, kw_args=dict(a_min=None, a_max=0.5)),
        ),
    ],
}

CLASSIFIERS = {
    'tree': lambda: DecisionTreeClassifier(random_state=0),
    'logistic_regression': lambda: LogisticRegression(max_iter=1000),
    'svc': lambda: SVC(),
}


def _tmp_dir(tmp_root_dir: Path, test: str, *params: tuple[str, str]):
    return fs_mkdir(tmp_root_dir, [('test', test), *params])


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('classifier', CLASSIFIERS.keys())
@pytest.mark.parametrize('preprocessing', PREPROCESSING.keys())
def test_pipeline_of_classifier(
    tmp_root_dir: Path,
    preprocessing: str,
    classifier: str,
    language: str,
    template: str,
):
    """Compare the predictions of a ported pipeline with scikit-learn."""
    x, y = load_iris(return_X_y=True)
    if preprocessing == 'imputer_with_indicator':
        x[::5, 1] = np.nan
    steps = PREPROCESSING[preprocessing]() + [
        ('clf', CLASSIFIERS[classifier]())
    ]
    pipeline = SklearnPipeline(steps).fit(x, y)
    if not Pipeline.supports(pipeline, language, template):
        pytest.skip(f'The template `{template}` is not supported.')

    est = Porter(pipeline, language=language, template=template)
    assert isinstance(est.ported, Pipeline)
    test_x = np.vstack((x, dataset_uniform_x(x)))
    if preprocessing in ('all_transformers', 'imputer_with_indicator'):
        test_x[::7, 0] = np.nan
    tmp_dir = _tmp_dir(
        tmp_root_dir,
        'pipeline_of_classifier',
        ('preprocessing', preprocessing),
        ('classifier', classifier),
        ('language', language),
        ('template', template),
    )
    y_pred, y_proba = est.predict(
        test_x, directory=tmp_dir, delete_created_files=False
    )
    # The ported transformers pass the transformed features without any
    # loss of precision to the ported classifier:
    np.testing.assert_array_equal(y_pred, pipeline.predict(test_x))
    if est.can(language, template, 'predict_proba'):
        assert isinstance(y_proba, np.ndarray)
        np.testing.assert_allclose(
            y_proba, pipeline.predict_proba(test_x), rtol=1e-12, atol=1e-15
        )
    assert est.test(test_x) == 1.0


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_pipeline_of_regressor(
    tmp_root_dir: Path, language: str, template: str
):
    """Compare the predictions of a ported pipeline with scikit-learn."""
    x, y = load_diabetes(return_X_y=True)
    pipeline = SklearnPipeline(
        [('scaler', StandardScaler()), ('reg', LinearRegression())]
    ).fit(x, y)
    if not Pipeline.supports(pipeline, language, template):
        pytest.skip(f'The template `{template}` is not supported.')

    est = Porter(pipeline, language=language, template=template)
    test_x = np.vstack((x, dataset_uniform_x(x)))
    tmp_dir = _tmp_dir(
        tmp_root_dir,
        'pipeline_of_regressor',
        ('language', language),
        ('template', template),
    )
    y_pred, _ = est.predict(
        test_x, directory=tmp_dir, delete_created_files=False
    )
    # The sum of the products can differ by a few ULP from NumPy:
    np.testing.assert_allclose(y_pred, pipeline.predict(test_x), rtol=1e-12)
    assert est.test(test_x) == 1.0


@pytest.mark.parametrize('language', LANGUAGES)
def test_pipeline_with_a_single_sample(tmp_root_dir: Path, language: str):
    x, y = load_iris(return_X_y=True)
    pipeline = SklearnPipeline(
        [('scaler', StandardScaler()), ('tree', DecisionTreeClassifier())]
    ).fit(x, y)
    est = Porter(pipeline, language=language)
    tmp_dir = _tmp_dir(
        tmp_root_dir, 'pipeline_with_a_single_sample', ('language', language)
    )
    y_pred, y_proba = est.predict(
        x[0], directory=tmp_dir, delete_created_files=False
    )
    assert y_pred == pipeline.predict(x[:1])[0]
    np.testing.assert_array_equal(y_proba, pipeline.predict_proba(x[:1])[0])


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('candidate', CANDIDATES, ids=lambda x: x.name)
def test_pipeline_of_all_estimators(
    tmp_root_dir: Path, candidate: Candidate, language: str, template: str
):
    """Port a pipeline of a StandardScaler and each supported estimator."""
    if candidate in REGRESSORS:
        x, y = load_diabetes(return_X_y=True)
    else:
        x, y = load_iris(return_X_y=True)
    steps = [('scaler', StandardScaler()), ('est', candidate.create())]
    pipeline = SklearnPipeline(steps)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            if candidate in OUTLIER_DETECTORS:
                pipeline.fit(x)
            else:
                pipeline.fit(x, y)
    except ValueError as e:  # e.g. negative features of MultinomialNB
        pytest.skip(str(e))
    if not Pipeline.supports(pipeline, language, template):
        pytest.skip('Skip unsupported estimator/language/template combination')

    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    orig_est = pipeline[-1]
    if isinstance(orig_est, (KNeighborsClassifier, KNeighborsRegressor)):
        # Drop the samples with ties of the k-th and (k+1)-th neighbors:
        k = orig_est.n_neighbors
        dists, _ = orig_est.kneighbors(
            pipeline[:-1].transform(test_x), n_neighbors=k + 1
        )
        test_x = test_x[~np.isclose(dists[:, k - 1], dists[:, k])]

    est = Porter(pipeline, language=language, template=template)
    tmp_dir = _tmp_dir(
        tmp_root_dir,
        'pipeline_of_all_estimators',
        ('estimator', candidate.name),
        ('language', language),
        ('template', template),
    )
    try:
        y_pred, y_proba = est.predict(
            test_x, directory=tmp_dir, delete_created_files=False
        )
    except (exception.CodeTooLarge, exception.TooManyConstants) as e:
        pytest.skip(str(e))
    if candidate in REGRESSORS:
        np.testing.assert_allclose(
            y_pred.astype(float), pipeline.predict(test_x), rtol=1e-6, atol=1e-8
        )
    else:
        np.testing.assert_array_equal(y_pred, pipeline.predict(test_x))
        if est.can(language, template, 'predict_proba'):
            assert isinstance(y_proba, np.ndarray)
            np.testing.assert_allclose(
                y_proba, pipeline.predict_proba(test_x), rtol=0, atol=1e-5
            )


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('language', LANGUAGES)
def test_pipeline_with_samples_as_arguments(
    tmp_root_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    template: str,
):
    """Test the features of one sample as arguments of the pipeline."""
    cmds = []

    def run_and_record(cmd: list[str], **kwargs) -> CompletedProcess:
        cmds.append(cmd)
        return run(cmd, **kwargs)

    monkeypatch.setattr(BASE_MODULE, 'run', run_and_record)

    x, y = load_iris(return_X_y=True)
    pipeline = SklearnPipeline(
        [('scaler', StandardScaler()), ('tree', DecisionTreeClassifier())]
    ).fit(x, y)
    est = Porter(pipeline, language=language, template=template)
    tmp_dir = _tmp_dir(
        tmp_root_dir,
        'pipeline_with_samples_as_arguments',
        ('language', language),
        ('template', template),
    )
    y_pred, _ = est.predict(
        x[::15], directory=tmp_dir, delete_created_files=False
    )
    assert len(cmds) == 1  # one process predicts all samples
    assert isinstance(y_pred, np.ndarray)
    for features, y_true in zip(x[::15].tolist(), y_pred, strict=True):
        args = list(map(str, features))
        proc = run(
            cmds[0] + args, capture_output=True, encoding='utf-8', check=True
        )
        assert json.loads(proc.stdout)['predict'] == y_true
