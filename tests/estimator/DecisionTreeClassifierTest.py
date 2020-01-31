# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import pytest
import numpy as np

# scikit-learn
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from sklearn_porter import exceptions as exception
from tests.commons import Dataset, DATASETS
from tests.utils import dataset_uniform_x, dataset_generate_x, fs_mkdir
from tests.EstimatorTest import tmp_root_dir  # fixture


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
@pytest.mark.parametrize('max_depth', [5, 10, 100])
@pytest.mark.parametrize('max_leaf_nodes', [2, 3])
def test_estimator_decision_tree_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    max_depth: int,
    max_leaf_nodes: int,
):
    """Test and compare different DecisionTree classifiers."""
    orig_est = DecisionTreeClassifier(
        max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=1
    )

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
            ('dataset', dataset.name), ('language', language),
            ('template', template), ('max_depth', str(max_depth)),
            ('max_leaf_nodes', str(max_leaf_nodes))
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
    except Exception as e:
        pytest.fail('Unexpected exception ... ' + str(e))
    else:
        assert score == 1.
