# -*- coding: utf-8 -*-

from ..Template import Template


class Regressor(Template):

    def __init__(self, model, **kwargs):
        # pylint: disable=unused-argument
        super(Regressor, self).__init__(model, **kwargs)
        self.algorithm_type = 'regressor'
