# -*- coding: utf-8 -*-

from typing import Union, Tuple

from sklearn.neural_network.multilayer_perceptron \
    import MLPRegressor as MLPRegressorClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class MLPRegressor(EstimatorBase, EstimatorApiABC):
    """
    Extract model data and port a MLPRegressor regressor.

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """
    estimator = None  # type: MLPRegressorClass

    def __init__(self, estimator: MLPRegressorClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        self.meta_info = dict()
        self.model_data = dict()

    def port(
            self,
            method: Method,
            language: Language,
            template: Template,
            **kwargs
    ) -> Union[str, Tuple[str, str]]:
        """
        Port an estimator.

        Parameters
        ----------
        method : Method
            The required method.
        language : Language
            The required language.
        template : Template
            The required template.
        kwargs

        Returns
        -------
        The ported estimator.
        """

        super().check_arguments(method, language, template)

        converter = kwargs.get('converter')

        # Placeholders:
        placeholders = dict(
            class_name=kwargs.get('class_name'),
            method_name=kwargs.get('method_name'),
        )
        placeholders.update({  # merge all placeholders
            **self.model_data,
            **self.meta_info
        })

        # Load templates:
        temps = self._load_templates(language.value.KEY)

        return str(self.estimator)
