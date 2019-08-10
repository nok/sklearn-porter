# -*- coding: utf-8 -*-

from copy import deepcopy
from json import dumps, encoder
from logging import DEBUG
from typing import Optional, Tuple, Union

# scikit-learn
from sklearn.naive_bayes import GaussianNB as GaussianNBClass
from sklearn_porter.enums import Language, Method, Template

# sklearn-porter
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.exceptions import NotFittedEstimatorError
from sklearn_porter.utils import get_logger

L = get_logger(__name__)


class GaussianNB(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a GaussianNB classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_TEMPLATE = Template.ATTACHED
    DEFAULT_METHOD = Method.PREDICT

    SUPPORT = {
        Language.JAVA: {
            Template.ATTACHED: {Method.PREDICT, Method.PREDICT_PROBA},
            Template.EXPORTED: {Method.PREDICT, Method.PREDICT_PROBA},
        },
        Language.JS: {
            Template.ATTACHED: {Method.PREDICT, Method.PREDICT_PROBA},
            Template.EXPORTED: {Method.PREDICT, Method.PREDICT_PROBA},
        },
    }

    estimator = None  # type: GaussianNBClass

    def __init__(self, estimator: GaussianNBClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Is the estimator fitted?
        try:
            est.sigma_
        except AttributeError:
            raise NotFittedEstimatorError(self.estimator_name)

        self.meta_info = dict(
            n_features=len(est.sigma_[0]),
            n_classes=len(est.classes_),
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

        self.model_data = dict(
            priors=est.class_prior_.tolist(),
            sigmas=est.sigma_.tolist(),
            thetas=est.theta_.tolist(),
        )
        L.info('Model data (keys): {}'.format(self.model_data.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Model data: {}'.format(self.model_data))

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
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        priors_val = self.model_data.get('priors')
        priors_val_converted = list(map(converter, priors_val))
        priors_str = tpl_arr_1.render(
            type=tpl_double,
            name='priors',
            values=', '.join(priors_val_converted),
            n=len(priors_val_converted)
        )

        sigmas_val = self.model_data.get('sigmas')
        sigmas_str = tpl_arr_2.render(
            type=tpl_double,
            name='sigmas',
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    ) for v in sigmas_val
                )
            ),
            n=len(sigmas_val),
            m=len(sigmas_val[0])
        )

        thetas_val = self.model_data.get('thetas')
        thetas_str = tpl_arr_2.render(
            type=tpl_double,
            name='thetas',
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    ) for v in thetas_val
                )
            ),
            n=len(thetas_val),
            m=len(thetas_val[0])
        )

        plas.update(
            dict(
                priors=priors_str,
                sigmas=sigmas_str,
                thetas=thetas_str,
            )
        )

        tpl_class = tpls.get_template('attached.class')
        out_class = tpl_class.render(**plas)

        return out_class
