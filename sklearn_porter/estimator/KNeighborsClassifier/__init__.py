# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
from json import encoder, dumps
from textwrap import indent
from copy import deepcopy
from logging import DEBUG

from sklearn.neighbors.classification import KNeighborsClassifier \
    as KNeighborsClassifierClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.exceptions import NotSupportedYetError
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class KNeighborsClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a KNeighborsClassifier classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_METHOD = Method.PREDICT
    DEFAULT_TEMPLATE = Template.ATTACHED

    SUPPORT = {
        Language.JAVA: {Method.PREDICT: {Template.ATTACHED, Template.EXPORTED}},
        Language.JS: {Method.PREDICT: {Template.ATTACHED, Template.EXPORTED}},
    }

    estimator = None  # type: KNeighborsClassifierClass

    def __init__(self, estimator: KNeighborsClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        if est.weights != 'uniform':
            msg = 'Only `uniform` weights are supported.'
            raise NotSupportedYetError(msg)

        self.meta_info = dict(
            n_classes=len(est.classes_),
            n_templates=len(est._fit_X),  # pylint: disable=W0212
            n_features=len(est._fit_X[0]),  # pylint: disable=W0212
            metric=est.metric
        )
        L.info('Meta info (keys): {}'.format(
            self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

        self.model_data = dict(
            X=est._fit_X.tolist(),  # pylint: disable=W0212
            y=est._y.astype(int).tolist(),  # pylint: disable=W0212
            k=est.n_neighbors,  # number of relevant neighbors
            n=len(est.classes_),  # number of classes
            power=est.p
        )
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
        tpls = self._load_templates(language.value.KEY)

        # Export:
        if template == Template.EXPORTED:
            tpl_class = tpls.get('exported.class')
            out_class = tpl_class.format(**plas)
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_int = tpls.get('int')
        tpl_double = tpls.get('double')
        tpl_arr_1 = tpls.get('arr[]')
        tpl_arr_2 = tpls.get('arr[][]')
        tpl_in_brackets = tpls.get('in_brackets')
        tpl_indent = tpls.get('indent')

        # Make method:
        metric_name = 'attached.metric.' + self.meta_info.get('metric')
        tpl_metric = tpls.get(metric_name)
        plas.update(dict(dist_comp=tpl_metric))
        tpl_method = tpls.get('attached.method.predict')
        out_method = tpl_method.format(**plas)

        if language in (Language.JAVA, Language.JS):
            n_indents = 1
            out_method = indent(out_method, n_indents * tpl_indent)
            out_method = out_method[(n_indents * len(tpl_indent)):]

        x_val = self.model_data.get('X')
        x_str = tpl_arr_2.format(
            type=tpl_double,
            name='X',
            values=', '.join(list(tpl_in_brackets.format(', '.join(
                list(map(converter, v)))) for v in x_val)),
            n=len(x_val),
            m=len(x_val[0])
        )

        y_val = list(map(str, self.model_data['y']))
        y_str = tpl_arr_1.format(
            type=tpl_int,
            name='y',
            values=', '.join(y_val),
            n=len(y_val)
        )

        # Make class:
        tpl_class = tpls.get('attached.class')
        plas.update(dict(
            method=out_method,
            X=x_str,
            y=y_str,
            k=self.model_data.get('k'),
            n=self.model_data.get('n'),
            power=self.model_data.get('power'),
        ))
        out_class = tpl_class.format(**plas)

        return out_class
