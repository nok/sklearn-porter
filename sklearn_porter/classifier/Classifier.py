# -*- coding: utf-8 -*-

from ..Template import Template


class Classifier(Template):

    def __init__(self, model, **kwargs):
        # pylint: disable=unused-argument
        super(Classifier, self).__init__(model, **kwargs)
        self.algorithm_type = 'classifier'
