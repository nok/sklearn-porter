# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import pytest
import numpy as np

# scikit-learn
from sklearn.neighbors.classification import \
    KNeighborsClassifier as KNeighborsClassifierClass

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from sklearn_porter import exceptions as exception
from tests.commons import Dataset, DATASETS
from tests.utils import dataset_uniform_x, dataset_generate_x, fs_mkdir
from tests.EstimatorTest import tmp_root_dir  # fixture


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('n_neighbors', [1, 2, 3, 5])
def test_estimator_k_neighbors_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_neighbors: int,
):
    """Test and compare different k-Neighbors classifiers."""
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPClassifier
    except ImportError:
        pass
    else:
        orig_est = KNeighborsClassifierClass(n_neighbors=n_neighbors)

        if not can(orig_est, language, template, 'predict'):
            pytest.skip(
                'Skip unsupported estimator/'
                'language/template combination'
            )

        # Estimator:
        x, y = dataset.data, dataset.target
        orig_est.fit(X=x, y=y)
        est = Estimator(orig_est)

        # Samples:
        test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

        # Directory:
        tmp_dir = fs_mkdir(
            tmp_root_dir, [
                ('test', 'estimator_decision_tree_classifier'),
                ('dataset', dataset.name),
                ('language', language),
                ('template', template),
                ('n_neighbors', str(n_neighbors)),
            ]
        )

        try:
            score = est.test(
                test_x,
                language=language,
                template=template,
                directory=tmp_dir,
                delete_created_files=False
            )
        except exception.CodeTooLarge:
            msg = 'Code too large for the combination: ' \
                  'language: {}, template: {}, dataset: {}' \
                  ''.format(language, template, dataset.name)
            warnings.warn(msg)
        except:
            pytest.fail('Unexpected exception ...')
        else:
            assert score == 1.
