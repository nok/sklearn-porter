# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
from textwrap import indent
from json import dumps, encoder

from sklearn.svm.classes import SVC as SVCClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.exceptions import NotFittedEstimatorError
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class SVC(EstimatorBase, EstimatorApiABC):
    """
    Extract model data and port a SVC classifier.

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_METHOD = Method.PREDICT
    DEFAULT_TEMPLATE = Template.ATTACHED

    SUPPORT = {
        Language.C: {Method.PREDICT: {Template.ATTACHED}},
        Language.JAVA: {Method.PREDICT: {Template.ATTACHED, Template.EXPORTED}},
        Language.JS: {Method.PREDICT: {Template.ATTACHED}},
        Language.PHP: {Method.PREDICT: {Template.ATTACHED}},
        Language.RUBY: {Method.PREDICT: {Template.ATTACHED}},
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
            msg = 'The kernel type is not supported.'
            raise ValueError(msg)

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

        self.meta_info = dict(
            n_classes=len(est.classes_),
            n_features=len(est.support_vectors_[0]),
            n_weights=len(self.model_data['weights']),
            n_vectors=len(self.model_data['vectors']),
            n_coeffs=len(self.model_data['coeffs']),
            n_inters=len(self.model_data['inters']),
        )

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
        placeholders.update(self.meta_info)

        # Load templates:
        tpls = self._load_templates(language.value.KEY)

        if template == Template.EXPORTED:
            output = str(tpls.get('exported.class').format(**placeholders))
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, sort_keys=True)
            return output, model_data

        # Pick templates:
        tpl_int = tpls.get('int')
        tpl_double = tpls.get('double')
        tpl_arr_1 = tpls.get('arr[]')
        tpl_arr_2 = tpls.get('arr[][]')
        tpl_in_brackets = tpls.get('in_brackets')
        tpl_indent = tpls.get('indent')

        # Convert weights:
        weights_val = list(map(lambda x: str(int(x)),
                               self.model_data['weights']))
        weights_str = tpl_arr_1.format(
            type=tpl_int,
            name='weights',
            values=', '.join(weights_val),
            n=len(weights_val)
        )

        # Convert vectors:
        vectors_val = self.model_data['vectors']
        vectors_str = tpl_arr_2.format(
            type=tpl_double,
            name='vectors',  # convert 2D lists to a string `{{1, 2, 3}, {...}}`
            values=', '.join(list(tpl_in_brackets.format(', '.join(list(map(converter, v)))) for v in vectors_val)),
            n=len(vectors_val),
            m=len(vectors_val[0])
        )

        # Convert coefficients:
        coeffs_val = self.model_data['coeffs']
        coeffs_str = tpl_arr_2.format(
            type=tpl_double,
            name='coeffs',
            values=', '.join(list(tpl_in_brackets.format(', '.join(list(map(converter, v)))) for v in coeffs_val)),
            n=len(coeffs_val),
            m=len(coeffs_val[0])
        )

        # Convert interceptions:
        inters_val = self.model_data['inters']
        inters_str = tpl_arr_1.format(
            type=tpl_double,
            name='inters',
            values=', '.join(list(map(converter, inters_val))),
            n=len(self.model_data['inters'])
        )

        # Convert kernel:
        kernel_val = self.model_data['kernel']
        kernel_str = str(kernel_val)[0] \
            if language == Language.C else str(kernel_val)

        # Convert gamma:
        gamma_val = self.model_data['gamma']
        gamma_str = converter(gamma_val)

        # Convert coefficient:
        coef0_val = self.model_data['coef0']
        coef0_str = converter(coef0_val)

        # Convert degree:
        degree_val = self.model_data['degree']
        degree_str = converter(degree_val)

        placeholders.update(dict(
            weights=weights_str,
            vectors=vectors_str,
            coeffs=coeffs_str,
            inters=inters_str,
            kernel=kernel_str,
            gamma=gamma_str,
            coef0=coef0_str,
            degree=degree_str
        ))

        # Make method:
        tpl_method = tpls.get('attached.method')
        n_indents = 1 if language in [
            Language.JAVA,
            Language.JS,
            Language.PHP,
            Language.RUBY
        ] else 0
        tpl_method = indent(tpl_method, n_indents * tpl_indent)
        tpl_method = tpl_method[(n_indents * len(tpl_indent)):]
        out_method = tpl_method.format(**placeholders)
        placeholders.update(dict(method=out_method))

        # Make class:
        tpl_class = tpls.get('attached.class')
        out_class = tpl_class.format(**placeholders)

        return out_class
