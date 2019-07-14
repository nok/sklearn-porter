# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
from json import encoder, dumps
from textwrap import indent
from copy import deepcopy
from logging import DEBUG

from sklearn.svm.classes import LinearSVC as LinearSVCClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.exceptions import NotFittedEstimatorError
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class LinearSVC(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a LinearSVC classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_TEMPLATE = Template.ATTACHED
    DEFAULT_METHOD = Method.PREDICT

    SUPPORT = {
        Language.C: {
            Template.ATTACHED: {Method.PREDICT, },
        },
        Language.GO: {
            Template.ATTACHED: {Method.PREDICT, },
        },
        Language.JAVA: {
            Template.ATTACHED: {Method.PREDICT, },
            Template.EXPORTED: {Method.PREDICT, },
        },
        Language.JS: {
            Template.ATTACHED: {Method.PREDICT, },
        },
        Language.PHP: {
            Template.ATTACHED: {Method.PREDICT, },
        },
        Language.RUBY: {
            Template.ATTACHED: {Method.PREDICT, },
        }
    }

    estimator = None  # type: LinearSVCClass

    def __init__(self, estimator: LinearSVCClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Is the estimator fitted?
        try:
            est.coef_
        except AttributeError:
            raise NotFittedEstimatorError(self.estimator_name)

        self.meta_info = dict(
            n_features=len(est.coef_[0]),
            n_classes=len(est.classes_)
        )
        L.info('Meta info (keys): {}'.format(
            self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

        is_binary = self.meta_info['n_classes'] == 2
        if is_binary:
            self.model_data = dict(
                coeffs=est.coef_[0].tolist(),
                inters=est.intercept_[0]
            )
        else:
            self.model_data = dict(
                coeffs=est.coef_.tolist(),
                inters=est.intercept_.tolist()
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

        is_binary = self.meta_info['n_classes'] == 2
        variant = 'binary' if is_binary else 'multi'

        # Export:
        if template == Template.EXPORTED:
            tpl_name = 'exported.' + variant + '.class'
            tpl_class = tpls.get(tpl_name)
            out_class = tpl_class.format(**plas)
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_init = tpls.get('init')
        tpl_double = tpls.get('double')
        tpl_arr_1 = tpls.get('arr[]')
        tpl_arr_2 = tpls.get('arr[][]')
        tpl_in_brackets = tpls.get('in_brackets')
        tpl_indent = tpls.get('indent')

        if is_binary:

            inters_val = converter(self.model_data.get('inters'))
            inters_str = tpl_init.format(
                type=tpl_double,
                name='inters',
                value=inters_val
            )

            coeffs_val = list(map(converter, self.model_data.get('coeffs')))
            coeffs_str = tpl_arr_1.format(
                type=tpl_double,
                name='coeffs',
                value=', '.join(coeffs_val),
                n=len(coeffs_val)
            )

        else:  # is multi

            inters_val = list(map(converter, self.model_data.get('inters')))
            inters_str = tpl_arr_1.format(
                type=tpl_double,
                name='inters',
                values=', '.join(inters_val),
                n=len(inters_val)
            )

            coeffs_val = self.model_data.get('coeffs')
            coeffs_str = tpl_arr_2.format(
                type=tpl_double,
                name='coeffs',
                values=', '.join(list(tpl_in_brackets.format(', '.join(
                    list(map(converter, v)))) for v in coeffs_val)),
                n=len(coeffs_val),
                m=len(coeffs_val[0])
            )

        plas.update(dict(
            coeffs=coeffs_str,
            inters=inters_str,
        ))

        # Make method:
        n_indents = 1 if language is Language.JAVA else 0
        tpl_method = tpls.get('attached.' + variant + '.method')
        tpl_method = indent(tpl_method, n_indents * tpl_indent)
        tpl_method = tpl_method[(n_indents * len(tpl_indent)):]
        out_method = tpl_method.format(**plas)
        plas.update(dict(method=out_method))

        # Make class:
        if language in (Language.JAVA, Language.GO):
            tpl_class_head = tpls.get('attached.' + variant + '.class')
            tpl_class_head = indent(tpl_class_head, n_indents * tpl_indent)
            tpl_class_head = tpl_class_head[(n_indents * len(tpl_indent)):]
            out_class_head = tpl_class_head.format(**plas)
            plas.update(dict(class_head=out_class_head))
        tpl_class = tpls.get('attached.class')
        out_class = tpl_class.format(**plas)

        return out_class
