# -*- coding: utf-8 -*-

from copy import deepcopy
from json import dumps, encoder
from typing import Optional, Tuple, Union

from loguru import logger as L

# scikit-learn
from sklearn.svm.classes import LinearSVC as LinearSVCClass

# sklearn-porter
from sklearn_porter.enums import Language, Method, Template
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.exceptions import NotFittedEstimatorError


class LinearSVC(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a LinearSVC classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_TEMPLATE = Template.ATTACHED
    DEFAULT_METHOD = Method.PREDICT

    SUPPORT = {
        Language.C: {
            Template.ATTACHED: {Method.PREDICT},
        },
        Language.GO: {
            Template.ATTACHED: {Method.PREDICT},
        },
        Language.JAVA: {
            Template.ATTACHED: {Method.PREDICT},
            Template.EXPORTED: {Method.PREDICT},
        },
        Language.JS: {
            Template.ATTACHED: {Method.PREDICT},
            Template.EXPORTED: {Method.PREDICT},
        },
        Language.PHP: {
            Template.ATTACHED: {Method.PREDICT},
            Template.EXPORTED: {Method.PREDICT},
        },
        Language.RUBY: {
            Template.ATTACHED: {Method.PREDICT},
            Template.EXPORTED: {Method.PREDICT},
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
            n_classes=len(est.classes_),
            is_binary=len(est.classes_) == 2
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        L.opt(lazy=True).debug('Meta info: {}'.format(self.meta_info))

        if self.meta_info['is_binary']:
            self.model_data = dict(
                coeffs=est.coef_[0].tolist(), inters=est.intercept_[0].tolist()
            )
        else:
            self.model_data = dict(
                coeffs=est.coef_.tolist(), inters=est.intercept_.tolist()
            )
        L.info('Model data (keys): {}'.format(self.model_data.keys()))
        L.opt(lazy=True).debug('Model data: {}'.format(self.model_data))

    def port(
        self,
        language: Optional[Language] = None,
        template: Optional[Template] = None,
        to_json: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, str]]:
        """
        Port an estimator.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        to_json : bool (default: False)
            Return the result as JSON string.
        kwargs

        Returns
        -------
        The ported estimator.
        """
        method, language, template = self.check(
            language=language, template=template
        )

        # Arguments:
        kwargs.setdefault('method_name', method.value)
        converter = kwargs.get('converter')

        # Placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(
            dict(
                class_name=kwargs.get('class_name'),
                method_name=kwargs.get('method_name'),
                to_json=to_json,
            )
        )
        plas.update(self.meta_info)

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Export:
        if template == Template.EXPORTED:
            tpl_name = 'exported.class'
            tpl_class = tpls.get_template(tpl_name)
            out_class = tpl_class.render(**plas)
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_init = tpls.get_template('init')
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        if self.meta_info['is_binary']:

            inters_val = converter(self.model_data.get('inters'))
            inters_str = tpl_init.render(
                type=tpl_double, name='inters', value=inters_val
            )

            coeffs_val = list(map(converter, self.model_data.get('coeffs')))
            coeffs_str = tpl_arr_1.render(
                type=tpl_double,
                name='coeffs',
                values=', '.join(coeffs_val),
                n=len(coeffs_val)
            )

        else:  # is multi

            inters_val = list(map(converter, self.model_data.get('inters')))
            inters_str = tpl_arr_1.render(
                type=tpl_double,
                name='inters',
                values=', '.join(inters_val),
                n=len(inters_val)
            )

            coeffs_val = self.model_data.get('coeffs')
            coeffs_str = tpl_arr_2.render(
                type=tpl_double,
                name='coeffs',
                values=', '.join(
                    list(
                        tpl_in_brackets.render(
                            value=', '.join(list(map(converter, v)))
                        ) for v in coeffs_val
                    )
                ),
                n=len(coeffs_val),
                m=len(coeffs_val[0])
            )

        plas.update(dict(
            coeffs=coeffs_str,
            inters=inters_str,
        ))

        tpl_class = tpls.get_template('attached.class')
        out_class = tpl_class.render(**plas)
        return out_class
