# -*- coding: utf-8 -*-

from ..SVC import SVC


class NuSVC(SVC):
    """
    See also
    --------
    sklearn.svm.NuSVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.NuSVC.html
    """
    def __init__(self, model, target_language='java', target_method='predict', **kwargs):
        super(NuSVC, self).__init__(model, target_language=target_language, target_method=target_method, **kwargs)
