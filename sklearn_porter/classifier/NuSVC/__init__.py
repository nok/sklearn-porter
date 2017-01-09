# -*- coding: utf-8 -*-

from ..SVC import SVC


class NuSVC(SVC):
    """
    See also
    --------
    sklearn.svm.NuSVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.NuSVC.html
    """

    def __init__(
            self, language='java', method_name='predict', class_name='Tmp'):
        super(NuSVC, self).__init__(language, method_name, class_name)
