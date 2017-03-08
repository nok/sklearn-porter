# -*- coding: utf-8 -*-

from ..RandomForestClassifier import RandomForestClassifier


class ExtraTreesClassifier(RandomForestClassifier):
    """
    See also
    --------
    sklearn.ensemble.ExtraTreesClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    """

    def __init__(self, model, target_language='java', target_method='predict', **kwargs):
        super(ExtraTreesClassifier, self).__init__(model, target_language=target_language, target_method=target_method, **kwargs)
