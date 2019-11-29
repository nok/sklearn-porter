# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union
import warnings

import pytest
import numpy as np

# scikit-learn
from sklearn.svm import SVC

# sklearn-porter
from sklearn_porter.Estimator import Estimator, can
from sklearn_porter import exceptions as exception
from tests.commons import Dataset, DATASETS
from tests.utils import dataset_uniform_x, dataset_generate_x, fs_mkdir
from tests.EstimatorTest import tmp_root_dir  # fixture


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', ['c', 'go', 'java', 'js', 'php', 'ruby'])
@pytest.mark.parametrize('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
@pytest.mark.parametrize(
    'gamma', [0.001, 0.01, 'auto'], ids=['0_001', '0_01', 'auto']
)
@pytest.mark.parametrize('dataset', DATASETS, ids=lambda x: x.name)
def test_estimator_svc(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    kernel: str,
    gamma: Union[str, float],
):
    """Test and compare different MLP classifiers."""
    try:
        from sklearn.neural_network.multilayer_perceptron import MLPClassifier
    except ImportError:
        pass
    else:
        orig_est = SVC(
            kernel=kernel,
            gamma=gamma,
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
        if not isinstance(gamma, str):
            gamma = str(gamma).replace('.', '_')
        tmp_dir = fs_mkdir(
            tmp_root_dir, [
                ('test', 'estimator_svc'), ('dataset', dataset.name),
                ('language', language), ('template', template),
                ('kernel', kernel), ('gamma', gamma)
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
        except exception.NotSupportedYetError as e:
            if 'template' == 'precomputed':
                msg = 'The passed kernel `precomputed` is not supported.'
                assert msg in str(e)
        except exception.CodeTooLarge:
            msg = 'Code too large for the combination: ' \
                  'language: {}, template: {}, dataset: {}' \
                  ''.format(language, template, dataset.name)
            warnings.warn(msg)
        except:
            pytest.fail('Unexpected exception ...')
        else:
            assert score == 1.
