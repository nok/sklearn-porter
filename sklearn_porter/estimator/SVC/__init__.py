# -*- coding: utf-8 -*-

from copy import deepcopy
from json import dumps, encoder
from logging import DEBUG
from typing import Optional, Tuple, Union

# scikit-learn
from sklearn.svm.classes import SVC as SVCClass

# sklearn-porter
from sklearn_porter.enums import Language, Method, Template
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.exceptions import (
    NotFittedEstimatorError, NotSupportedYetError
)
from sklearn_porter.utils import get_logger

L = get_logger(__name__)


class SVC(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a SVC classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_TEMPLATE = Template.ATTACHED
    DEFAULT_METHOD = Method.PREDICT

    SUPPORT = {
        Language.C: {
            Template.ATTACHED: {
                Method.PREDICT,
            },
        },
        Language.JAVA: {
            Template.ATTACHED: {
                Method.PREDICT,
            },
            Template.EXPORTED: {
                Method.PREDICT,
            },
        },
        Language.JS: {
            Template.ATTACHED: {
                Method.PREDICT,
            },
        },
        Language.PHP: {
            Template.ATTACHED: {
                Method.PREDICT,
            },
        },
        Language.RUBY: {
            Template.ATTACHED: {
                Method.PREDICT,
            },
        },
    }

    estimator = None  # type: SVCClass

    def __init__(self, estimator: SVCClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias
        params = est.get_params()

        # Is the estimator fitted?
        try:
            est.support_vectors_
        except AttributeError:
            raise NotFittedEstimatorError(self.estimator_name)

        # Check kernel type:
        supported_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if params['kernel'] not in supported_kernels:
            msg = 'The passed kernel `{}` is not supported.'
            msg = msg.format(params['kernel'])
            raise NotSupportedYetError(msg)

        # Check gamma value:
        gamma = params['gamma']
        if str(gamma).startswith('auto') or str(gamma).startswith('scale'):
            gamma = 1. / len(est.support_vectors_[0])

        self.model_data = dict(
            weights=est.n_support_.tolist(),
            vectors=est.support_vectors_.tolist(),
            coeffs=est.dual_coef_.tolist(),
            inters=est._intercept_.tolist(),
            kernel=params['kernel'],
            gamma=gamma,
            coef0=params['coef0'],
            degree=params['degree'],
        )
        L.info('Model data (keys): {}'.format(self.model_data.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Model data: {}'.format(self.model_data))

        self.meta_info = dict(
            n_classes=len(est.classes_),
            n_features=len(est.support_vectors_[0]),
            n_weights=len(self.model_data.get('weights')),
            n_vectors=len(self.model_data.get('vectors')),
            n_coeffs=len(self.model_data.get('coeffs')),
            n_inters=len(self.model_data.get('inters')),
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

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
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_int = tpls.get_template('int').render()
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        # Convert weights:
        weights_val = list(
            map(lambda x: str(int(x)), self.model_data.get('weights'))
        )
        weights_str = tpl_arr_1.render(
            type=tpl_int,
            name='weights',
            values=', '.join(weights_val),
            n=len(weights_val)
        )

        # Convert vectors:
        vectors_val = self.model_data.get('vectors')
        vectors_str = tpl_arr_2.render(
            type=tpl_double,
            name='vectors',  # convert 2D lists to a string `{{1, 2, 3}, {...}}`
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    ) for v in vectors_val
                )
            ),
            n=len(vectors_val),
            m=len(vectors_val[0])
        )

        # Convert coefficients:
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

        # Convert interceptions:
        inters_val = self.model_data.get('inters')
        inters_str = tpl_arr_1.render(
            type=tpl_double,
            name='inters',
            values=', '.join(list(map(converter, inters_val))),
            n=len(self.model_data.get('inters'))
        )

        # Convert kernel:
        kernel_val = self.model_data.get('kernel')
        kernel_str = str(kernel_val)[0] \
            if language == Language.C else str(kernel_val)

        # Convert gamma:
        gamma_val = self.model_data.get('gamma')
        gamma_str = converter(gamma_val)

        # Convert coefficient:
        coef0_val = self.model_data.get('coef0')
        coef0_str = converter(coef0_val)

        # Convert degree:
        degree_val = self.model_data.get('degree')
        degree_str = converter(degree_val)

        plas.update(
            dict(
                weights=weights_str,
                vectors=vectors_str,
                coeffs=coeffs_str,
                inters=inters_str,
                kernel=kernel_str,
                gamma=gamma_str,
                coef0=coef0_str,
                degree=degree_str
            )
        )

        # Make class:
        tpl_class = tpls.get_template('attached.class')
        out_class = tpl_class.render(**plas)
        return out_class
