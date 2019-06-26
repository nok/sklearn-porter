# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
from copy import deepcopy
from logging import DEBUG

from sklearn.neighbors.classification import KNeighborsClassifier \
    as KNeighborsClassifierClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class KNeighborsClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a KNeighborsClassifier classifier."""

    estimator = None  # type: KNeighborsClassifierClass

    def __init__(self, estimator: KNeighborsClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        self.meta_info = dict()
        L.info('Meta info (keys): {}'.format(
            self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

        self.model_data = dict()
        L.info('Model data (keys): {}'.format(
            self.model_data.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Model data: {}'.format(self.model_data))

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

        # Arguments:
        kwargs.setdefault('method_name', method.value)
        converter = kwargs.get('converter')

        # Placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(dict(
            class_name=kwargs.get('class_name'),
            method_name=kwargs.get('method_name'),
        ))
        plas.update(self.meta_info)

        # Templates:
        temps = self._load_templates(language.value.KEY)

        return str(self.estimator)
