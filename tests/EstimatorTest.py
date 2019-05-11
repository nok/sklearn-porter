# -*- coding: utf-8 -*-

import pytest

from sklearn_porter.Estimator import Estimator

import numpy as np
import random as rd

np.random.seed(0)
rd.seed(0)


@pytest.fixture(scope='module')
def sklearn_version():
    from sklearn import __version__ as sklearn_ver
    return tuple(map(int, str(sklearn_ver).split('.')))


def test_extraction_from_pipeline(sklearn_version):
    """Test the extraction of an estimator from a pipeline."""
    if sklearn_version[:2] >= (0, 15):
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline

        svc = SVC()
        pipeline = Pipeline([('SVM', svc)])
        est = Estimator(pipeline)

        assert isinstance(svc, SVC)
        assert isinstance(est.estimator, SVC)
