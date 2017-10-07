# -*- coding: utf-8 -*-

from sklearn_porter.estimator.classifier.RandomForestClassifier \
    import RandomForestClassifier


class ExtraTreesClassifier(RandomForestClassifier):
    """
    See also
    --------
    sklearn.ensemble.ExtraTreesClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    """

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param estimator : AdaBoostClassifier
            An instance of a trained ExtraTreesClassifier estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(ExtraTreesClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)
