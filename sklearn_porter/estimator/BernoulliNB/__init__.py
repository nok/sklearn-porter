# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
from json import encoder, dumps
from copy import deepcopy
from numpy import exp, log
from logging import DEBUG

from sklearn.naive_bayes import BernoulliNB as BernoulliNBClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class BernoulliNB(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a BernoulliNB classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_METHOD = Method.PREDICT
    DEFAULT_TEMPLATE = Template.ATTACHED

    SUPPORT = {
        Language.JAVA: {Method.PREDICT: {Template.ATTACHED, Template.EXPORTED}},
        Language.JS: {Method.PREDICT: {Template.ATTACHED, }},
    }

    estimator = None  # type: BernoulliNBClass

    def __init__(self, estimator: BernoulliNBClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        self.meta_info = dict(
            n_features=len(est.feature_log_prob_[0]),
            n_classes=len(est.classes_),
        )
        L.info('Meta info (keys): {}'.format(
            self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

        negatives = log(1 - exp(est.feature_log_prob_))
        deltas = (est.feature_log_prob_ - negatives).T

        self.model_data = dict(
            priors=est.class_log_prior_.tolist(),
            negatives=negatives.tolist(),
            deltas=deltas.tolist()
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

        # Export template:
        if language == Language.JAVA and method == Template.EXPORTED:
            output = str(tpls.get('exported.class').format(**plas))
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return output, model_data

        # Pick templates:
        tpl_double = tpls.get('double')
        tpl_arr_1 = tpls.get('arr[]')
        tpl_arr_2 = tpls.get('arr[][]')
        tpl_in_brackets = tpls.get('in_brackets')

        priors_val = self.model_data.get('priors')
        priors_val_converted = list(map(converter, priors_val))
        priors_str = tpl_arr_1.format(
            type=tpl_double,
            name='priors',
            values=', '.join(priors_val_converted),
            n=len(priors_val)
        )

        deltas_val = self.model_data.get('deltas')
        deltas_str = tpl_arr_2.format(
            type=tpl_double,
            name='deltas',
            values=', '.join(list(tpl_in_brackets.format(', '.join(
                list(map(converter, v)))) for v in deltas_val)),
            n=len(deltas_val),
            m=len(deltas_val[0])
        )

        negatives_val = self.model_data.get('negatives')
        negatives_str = tpl_arr_2.format(
            type=tpl_double,
            name='negatives',
            values=', '.join(list(tpl_in_brackets.format(', '.join(
                list(map(converter, v)))) for v in negatives_val)),
            n=len(negatives_val),
            m=len(negatives_val[0])
        )

        plas.update(dict(
            priors=priors_str,
            deltas=deltas_str,
            negatives=negatives_str,
        ))

        tpl_method = tpls.get('attached.method.predict')
        out_method = tpl_method.format(**plas)

        plas.update(dict(method=out_method))

        tpl_class = tpls.get('attached.class')
        out_class = tpl_class.format(**plas)

        return out_class
