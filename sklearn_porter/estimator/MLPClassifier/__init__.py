from copy import deepcopy
from json import dumps, encoder
from typing import Tuple, Union, Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.neural_network import \
    MLPClassifier as MLPClassifierClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class MLPClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a MLPClassifier classifier."""

    SKLEARN_URL = 'sklearn.neural_network.MLPClassifier.html'

    DEFAULT_LANGUAGE = enum.Language.JAVA
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.JAVA: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
    }

    estimator = None  # type: MLPClassifierClass

    def __init__(self, estimator: MLPClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Check output function:
        try:
            est.out_activation_
        except AttributeError:
            raise exception.NotFittedEstimatorError(self.estimator_name)

        # Architecture:
        n_inputs = len(est.coefs_[0])
        n_outputs = est.n_outputs_
        n_hidden_layers = est.hidden_layer_sizes
        if isinstance(n_hidden_layers, int):
            n_hidden_layers = [n_hidden_layers]
        n_hidden_layers = list(n_hidden_layers)
        layers = [n_inputs] + n_hidden_layers + [n_outputs]

        self.meta_info = dict(n_features=n_inputs, )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        L.opt(lazy=True).debug('Meta info: {}'.format(self.meta_info))

        self.model_data = dict(
            layers=list(map(int, layers[1:])),
            weights=list(map(np.ndarray.tolist, est.coefs_)),
            bias=list(map(np.ndarray.tolist, est.intercepts_)),
            hidden_activation=est.activation,
        )
        if self.estimator_name == 'MLPClassifier':
            self.model_data['output_activation'] = est.out_activation_

        L.info('Model data (keys): {}'.format(self.model_data.keys()))
        L.opt(lazy=True).debug('Model data: {}'.format(self.model_data))

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
        tpl_arr_3 = tpls.get_template('arr[][][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        # Convert layers:
        layers_val = self.model_data.get('layers')
        layers_str = tpl_arr_1.render(
            type=tpl_int,
            name='layers',
            values=', '.join(list(map(str, layers_val))),
            n=len(layers_val)
        )

        # Convert weights:
        weights_val = self.model_data.get('weights')
        weights_str = []
        for layer in weights_val:
            layer_weights = ', '.join(
                [
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, l)))
                    ) for l in layer
                ]
            )
            weights_str.append(tpl_in_brackets.render(value=layer_weights))
        weights_str = tpl_arr_3.render(
            type=tpl_double,
            name='weights',
            values=', '.join(weights_str),
            n=len(weights_val),
            m=len(weights_val[0]),
            k=len(weights_val[0][0]),
        )

        # Convert bias:
        bias_val = self.model_data.get('bias')
        bias_str = tpl_arr_2.render(
            type=tpl_double,
            name='bias',
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    ) for v in bias_val
                )
            ),
            n=len(bias_val),
            m=len(bias_val[0])
        )

        plas.update(
            dict(
                layers=layers_str,
                weights=weights_str,
                bias=bias_str,
                hidden_activation=self.model_data.get('hidden_activation'),
                output_activation=self.model_data.get('output_activation'),
            )
        )

        # Make class:
        tpl_class = tpls.get_template('attached.class')
        out_class = tpl_class.render(**plas)
        return out_class
