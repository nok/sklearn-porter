# -*- coding: utf-8 -*-

from sklearn_porter.estimator.classifier.SVC import SVC


class NuSVC(SVC):
    """
    See also
    --------
    sklearn.svm.NuSVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.NuSVC.html
    """
    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param estimator : AdaBoostClassifier
            An instance of a trained NuSVC estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(NuSVC, self).__init__(estimator, target_language=target_language,
                                    target_method=target_method, **kwargs)
