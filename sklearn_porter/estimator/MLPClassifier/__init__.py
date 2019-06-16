# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional

from sklearn.neural_network.multilayer_perceptron import MLPClassifier \
    as MLPClassifierClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class MLPClassifier(EstimatorBase, EstimatorApiABC):
    """
    Extract model data and port a MLPClassifier classifier.

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """
    estimator = None  # type: MLPClassifierClass

    def __init__(self, estimator: MLPClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        self.meta_info = dict()
        self.model_data = dict()

    def port(
            self,
            method: Optional[Method] = None,
            language: Optional[Language] = None,
            template: Optional[Template] = None,
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
        method, language, template = self.check(
            method=method, language=language, template=template)

        kwargs.setdefault('method_name', method.value)

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
