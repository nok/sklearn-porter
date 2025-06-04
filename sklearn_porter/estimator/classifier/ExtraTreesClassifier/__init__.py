# -*- coding: utf-8 -*-

from sklearn_porter.estimator.classifier.RandomForestClassifier \
    import RandomForestClassifier


class ExtraTreesClassifier(RandomForestClassifier):
    """
    See also
    --------
    sklearn.ensemble.ExtraTreesClassifier

    http://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.ExtraTreesClassifier.html
    """

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param estimator : ExtraTreesClassifier
            An instance of a trained ExtraTreesClassifier estimator.
        :param target_language : string, default: 'java'
            The target programming language.
        :param target_method : string, default: 'predict'
            The target method of the estimator.
        """
        super(ExtraTreesClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)
