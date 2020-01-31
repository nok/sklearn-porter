# -*- coding: utf-8 -*-

import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.Estimator import Estimator, can
from tests.commons import DATASETS, SKLEARN_VERSION, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

from tests.EstimatorTest import tmp_root_dir  # fixture


@pytest.mark.skipif(
    SKLEARN_VERSION[:2] < (0, 18), reason='requires scikit-learn >= v0.18'
)
@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('activation', ['relu', 'identity', 'tanh', 'logistic'])
@pytest.mark.parametrize(
    'hidden_layer_sizes', [15, [15, 5], [15, 10, 5]],
    ids=['15', '15_5', '15_10_5']
)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
def test_estimator_mlp_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    activation: str,
    hidden_layer_sizes: Union[int, List[int]],
):
    """Test and compare different MLP classifiers."""
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPClassifier
    except ImportError:
        pass
    else:
        orig_est = MLPClassifier(
            activation=activation,
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=.1,
            random_state=1,
            max_iter=10,
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
        hls = hidden_layer_sizes  # alias
        hls = '_'.join(
            [str(hls)] if isinstance(hls, int) else list(map(str, hls))
        )
        tmp_dir = fs_mkdir(
            tmp_root_dir, [
                ('test', 'estimator_mlp_classifier'), ('dataset', dataset.name),
                ('language', language), ('template', template),
                ('activation', activation), ('hidden_layer_sizes', hls)
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
