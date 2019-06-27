# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
from json import encoder, dumps
from copy import deepcopy
from logging import DEBUG
import numpy as np

from sklearn.neural_network.multilayer_perceptron import MLPClassifier \
    as MLPClassifierClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.exceptions import NotSupportedYetError
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class MLPClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a MLPClassifier classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_METHOD = Method.PREDICT
    DEFAULT_TEMPLATE = Template.ATTACHED

    SUPPORT = {
        Language.JAVA: {Method.PREDICT: {Template.ATTACHED, Template.EXPORTED}},
        Language.JS: {Method.PREDICT: {Template.ATTACHED, Template.EXPORTED}},
    }

    estimator = None  # type: MLPClassifierClass

    def __init__(self, estimator: MLPClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Check activation function:
        act_fn = est.activation
        supported_act_fns = ['relu', 'identity', 'tanh', 'logistic']
        if act_fn not in supported_act_fns:
            msg = 'The passed activation function `{}` is not supported yet.'
            msg = msg.format(act_fn)
            raise NotSupportedYetError(msg)

        # Check output function:
        out_fn = est.out_activation_
        supported_out_fns = ['softmax', 'logistic']
        if out_fn not in supported_out_fns:
            msg = 'The passed output function `{}` is not supported yet.'
            msg = msg.format(out_fn)
            raise NotSupportedYetError(msg)

        # Architecture:
        n_inputs = len(est.coefs_[0])
        n_outputs = est.n_outputs_
        n_hidden_layers = est.hidden_layer_sizes
        if isinstance(n_hidden_layers, int):
            n_hidden_layers = [n_hidden_layers]
        n_hidden_layers = list(n_hidden_layers)
        layers = [n_inputs] + n_hidden_layers + [n_outputs]

        self.meta_info = dict(
            n_features=n_inputs,
        )
        L.info('Meta info (keys): {}'.format(
            self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

        self.model_data = dict(
            layers=list(map(int, layers[1:])),
            weights=list(map(np.ndarray.tolist, est.coefs_)),
            bias=list(map(np.ndarray.tolist, est.coefs_)),
            hidden_activation=est.activation,
            output_activation=est.out_activation_,
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
        tpl_arr_3 = tpls.get('arr[][][]')
        tpl_in_brackets = tpls.get('in_brackets')

        # Convert layers:
        layers_val = self.model_data.get('layers')
        layers_str = tpl_arr_1.format(
            type=tpl_int,
            name='layers',
            values=', '.join(list(map(str, layers_val))),
            n=len(layers_val)
        )

        # Convert weights:
        weights_val = self.model_data.get('weights')
        weights_str = []
        for layer in weights_val:
            layer_weights = ', '.join([
                tpl_in_brackets.format(', '.join(
                    list(map(converter, l)))) for l in layer
            ])
            weights_str.append(tpl_in_brackets.format(layer_weights))
        weights_str = tpl_arr_3.format(
            type=tpl_double,
            name='weights',
            values=', '.join(weights_str),
            n=len(weights_val),
            m=len(weights_val[0]),
            k=len(weights_val[0][0]),
        )

        # Convert bias:
        bias_val = self.model_data.get('bias')
        bias_str = tpl_arr_2.format(
            type=tpl_double,
            name='bias',
            values=', '.join(list(tpl_in_brackets.format(
                ', '.join(list(map(converter, v)))) for v in bias_val)),
            n=len(bias_val),
            m=len(bias_val[0])
        )

        plas.update(dict(
            layers=layers_str,
            weights=weights_str,
            bias=bias_str,
            hidden_activation=self.model_data.get('hidden_activation'),
            output_activation=self.model_data.get('output_activation'),
        ))

        # Make class:
        tpl_class = tpls.get('attached.class')
        out_class = tpl_class.format(**plas)

        return out_class
