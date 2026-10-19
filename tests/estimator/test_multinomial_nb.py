from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

# scikit-learn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB, MultinomialNB

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, LANGUAGES, Dataset
from tests.utils import (
    dataset_generate_x,
    dataset_uniform_x,
    drop_class_ties,
    fs_mkdir,
)


def get_text_datasets() -> list[Dataset]:
    """
    Get the word counts and the tf-idf values of random documents, whose
    words have different frequencies in each class.
    """
    rng = np.random.default_rng(0)
    n_classes, n_features, n_samples = 3, 200, 300
    rates = rng.gamma(0.3, size=(n_classes, n_features))
    y = rng.integers(n_classes, size=n_samples)
    counts = rng.poisson(rates[y]).astype(float)
    tfidf = TfidfTransformer().fit_transform(counts).toarray()
    return [
        Dataset(name='word_counts', data=counts, target=y),
        Dataset(name='tfidf', data=tfidf, target=y),
    ]


TEXT_DATASETS = get_text_datasets()


def get_test_x(est, dataset: Dataset) -> np.ndarray:
    """Get the samples and a sample without features (e.g. empty text)."""
    x = dataset.data
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))
    test_x = drop_class_ties(est, test_x)
    return np.vstack((test_x, np.zeros((1, x.shape[1]))))


def check_estimator(
    tmp_root_dir: Path,
    orig_est: MultinomialNB | ComplementNB,
    dataset: Dataset,
    language: str,
    template: str,
    params_id: str,
):
    if not can(orig_est, language, template, 'predict'):
        pytest.skip('Skip unsupported estimator/language/template combination')

    # The text features are sparse, like the output of a vectorizer:
    is_text = dataset in TEXT_DATASETS
    x = csr_matrix(dataset.data) if is_text else dataset.data
    orig_est.fit(X=x, y=dataset.target)
    est = Estimator(orig_est)
    test_x = get_test_x(orig_est, dataset)
    if is_text:
        test_x = csr_matrix(test_x)

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_' + type(orig_est).__name__),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('params', params_id),
        ],
    )

    y_pred, y_proba = est.make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    np.testing.assert_allclose(
        y_proba, orig_est.predict_proba(test_x), rtol=0, atol=1e-5
    )


# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

MULTINOMIAL_NB_PARAMS = {
    'default': dict(),
    'small_alpha': dict(alpha=0.01),
    # `alpha=0` is replaced by `1e-10`:
    'no_force_alpha': dict(alpha=0.0, force_alpha=False),
    'no_fit_prior': dict(fit_prior=False),
    'class_prior': dict(),  # see below
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + TEXT_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', MULTINOMIAL_NB_PARAMS)
@pytest.mark.filterwarnings('ignore:alpha too small:UserWarning')
def test_estimator_multinomial_nb(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    params: str,
):
    """Test and compare MultinomialNB classifiers with different params."""
    kwargs = MULTINOMIAL_NB_PARAMS[params]
    if params == 'class_prior':
        n_classes = len(np.unique(dataset.target))
        priors = np.arange(1, n_classes + 1, dtype=float)
        kwargs = dict(class_prior=priors / priors.sum())
    orig_est = MultinomialNB(**kwargs)
    check_estimator(tmp_root_dir, orig_est, dataset, language, template, params)


COMPLEMENT_NB_PARAMS = {
    'default': dict(),
    'small_alpha': dict(alpha=0.01),
    'norm': dict(norm=True),
}


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + TEXT_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', COMPLEMENT_NB_PARAMS)
def test_estimator_complement_nb(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    params: str,
):
    """Test and compare ComplementNB classifiers with different params."""
    orig_est = ComplementNB(**COMPLEMENT_NB_PARAMS[params])
    check_estimator(tmp_root_dir, orig_est, dataset, language, template, params)


@pytest.mark.parametrize('template', ['attached', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('clazz', [MultinomialNB, ComplementNB])
def test_naive_bayes_with_a_single_class(
    tmp_root_dir: Path, clazz: type, language: str, template: str
):
    """Test the port of a classifier which was fitted on a single class."""
    dataset = DATASETS[1]  # iris
    x = dataset.data
    orig_est = clazz(class_prior=[0.5]).fit(X=x, y=np.zeros(len(x), int))
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'naive_bayes_with_a_single_class'),
            ('estimator', clazz.__name__),
            ('language', language),
            ('template', template),
        ],
    )
    test_x = x[::10]
    y_pred, y_proba = Estimator(orig_est).make(
        test_x,
        language=language,
        template=template,
        directory=tmp_dir,
        delete_created_files=False,
    )
    np.testing.assert_array_equal(y_pred, orig_est.predict(test_x))
    np.testing.assert_allclose(
        np.reshape(y_proba, (len(test_x), -1)),
        orig_est.predict_proba(test_x),
        rtol=0,
        atol=1e-5,
    )
