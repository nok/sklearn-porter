from pathlib import Path

import numpy as np
import pytest

# scikit-learn
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import Pipeline, Porter
from tests.commons import LANGUAGES
from tests.utils import dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

TEMPLATES = ['attached', 'combined', 'exported']

# The preprocessing steps (also two transformers of the same class):
PREPROCESSING = {
    'scaler': lambda: [('scaler', StandardScaler())],
    'two_scalers': lambda: [
        ('scaler', StandardScaler()),
        ('skipped', 'passthrough'),
        ('scaler2', StandardScaler(with_mean=False)),
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
    steps = PREPROCESSING[preprocessing]() + [
        ('clf', CLASSIFIERS[classifier]())
    ]
    pipeline = SklearnPipeline(steps).fit(x, y)
    if not Pipeline.supports(pipeline, language, template):
        pytest.skip(f'The template `{template}` is not supported.')

    est = Porter(pipeline, language=language, template=template)
    assert isinstance(est.ported, Pipeline)
    test_x = np.vstack((x, dataset_uniform_x(x)))
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
