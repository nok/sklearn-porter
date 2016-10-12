from RandomForestClassifier import RandomForestClassifier


class ExtraTreesClassifier(RandomForestClassifier):
    """
    See also
    --------
    sklearn.ensemble.ExtraTreesClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    """

    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(ExtraTreesClassifier, self).__init__(language, method_name, class_name)
