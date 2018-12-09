# -*- coding: utf-8 -*-

from sklearn_porter.Template import Template


class Classifier(Template):

    def __init__(self, estimator, **kwargs):
        # pylint: disable=unused-argument
        super(Classifier, self).__init__(estimator, **kwargs)
        self.estimator_type = 'classifier'
