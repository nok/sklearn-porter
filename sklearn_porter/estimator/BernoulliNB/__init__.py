# -*- coding: utf-8 -*-

from copy import deepcopy
from json import dumps, encoder
from logging import DEBUG
from typing import Optional, Tuple, Union

from numpy import exp, log

# scikit-learn
from sklearn.naive_bayes import BernoulliNB as BernoulliNBClass
from sklearn_porter.enums import Language, Method, Template

# sklearn-porter
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.exceptions import NotFittedEstimatorError
from sklearn_porter.utils import get_logger

L = get_logger(__name__)


class BernoulliNB(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a BernoulliNB classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_TEMPLATE = Template.ATTACHED
    DEFAULT_METHOD = Method.PREDICT

    SUPPORT = {
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
    }

    estimator = None  # type: BernoulliNBClass

    def __init__(self, estimator: BernoulliNBClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Is the estimator fitted?
        try:
            est.feature_log_prob_
        except AttributeError:
            raise NotFittedEstimatorError(self.estimator_name)

        self.meta_info = dict(
            n_features=len(est.feature_log_prob_[0]),
            n_classes=len(est.classes_),
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

        self.model_data = dict(
            class_log_prior=est.class_log_prior_.tolist(),
            feature_log_prob=est.feature_log_prob_.tolist(),
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

        priors_val = self.model_data.get('class_log_prior')
        priors_val_converted = list(map(converter, priors_val))
        priors_str = tpl_arr_1.render(
            type=tpl_double,
            name='priors',
            values=', '.join(priors_val_converted),
            n=len(priors_val)
        )

        probs_val = self.model_data.get('feature_log_prob')
        probs_str = tpl_arr_2.render(
            type=tpl_double,
            name='probs',
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    ) for v in probs_val
                )
            ),
            n=len(probs_val),
            m=len(probs_val[0])
        )

        plas.update(dict(
            priors=priors_str,
            probs=probs_str,
        ))

        tpl_class = tpls.get_template('attached.class')
        out_class = tpl_class.render(**plas)

        return out_class
