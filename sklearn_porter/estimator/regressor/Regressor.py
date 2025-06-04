# -*- coding: utf-8 -*-

from sklearn_porter.Template import Template


class Regressor(Template):

    def __init__(self, estimator, **kwargs):
        # pylint: disable=unused-argument
        super(Regressor, self).__init__(estimator, **kwargs)
        self.estimator_type = 'regressor'
