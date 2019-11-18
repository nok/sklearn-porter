# -*- coding: utf-8 -*-

from copy import deepcopy
from json import dumps, encoder
from typing import Optional, Tuple, Union, Callable

from loguru import logger as L

# scikit-learn
from sklearn.svm.classes import SVC as SVCClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class SVC(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a SVC classifier."""

    SKLEARN_URL = 'sklearn.svm.SVC.html'

    DEFAULT_LANGUAGE = enum.Language.JAVA
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
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
            raise exception.NotFittedEstimatorError(self.estimator_name)

        # Check kernel type:
        supported_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if params['kernel'] not in supported_kernels:
            msg = 'The passed kernel `{}` is not supported.'
            msg = msg.format(params['kernel'])
            raise exception.NotSupportedYetError(msg)

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
        L.opt(lazy=True).debug('Model data: {}'.format(self.model_data))

        self.meta_info = dict(
            n_classes=len(est.classes_),
            n_features=len(est.support_vectors_[0]),
            n_weights=len(self.model_data.get('weights')),
            n_vectors=len(self.model_data.get('vectors')),
            n_coeffs=len(self.model_data.get('coeffs')),
            n_inters=len(self.model_data.get('inters')),
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        L.opt(lazy=True).debug('Meta info: {}'.format(self.meta_info))

    def port(
        self,
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
        to_json: bool = False,
    ) -> Union[str, Tuple[str, str]]:
        """
        Port an estimator.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed estimator will be
            used, e.g. `DecisionTreeClassifier`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The ported estimator.
        """
        # Placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(dict(
            class_name=class_name,
            to_json=to_json,
        ))
        plas.update(self.meta_info)

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Make 'exported' variant:
        if template == enum.Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Make 'attached' variant:
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
            if language == enum.Language.C else str(kernel_val)

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
