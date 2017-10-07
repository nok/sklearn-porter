# -*- coding: utf-8 -*-

import numpy as np
from sklearn_porter.estimator.classifier.Classifier import Classifier

np.set_printoptions(precision=64)


class MLPClassifier(Classifier):
    """
    See also
    --------
    sklearn.neural_network.MLPClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.neural_network.MLPClassifier.html
    """

    SUPPORTED_METHODS = ['predict']

    # @formatter:off
    TEMPLATES = {
        'java': {
            'type':     '{0}',
            'arr':      '{{{0}}}',
            'new_arr':  'new {type}[{values}]',
            'arr[]':    '{type}[] {name} = {{{values}}};',
            'arr[][]':  '{type}[][] {name} = {{{values}}};',
            'arr[][][]': '{type}[][][] {name} = {{{values}}};',
            'indent':   '    ',
        },
        'js': {
            'type':     '{0}',
            'arr':      '[{0}]',
            'new_arr':  'new Array({values}).fill({fill_with})',
            'arr[]':    '{name} = [{values}];',
            'arr[][]':  '{name} = [{values}];',
            'arr[][][]': '{name} = [{values}];',
            'indent':   '    ',
        }
    }
    # @formatter:on

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param estimator : MLPClassifier
            An instance of a trained AdaBoostClassifier estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(MLPClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)
        self.estimator = estimator

        # Activation function ('identity', 'logistic', 'tanh' or 'relu'):
        self.hidden_activation = self.estimator.activation
        if self.hidden_activation not in self.hidden_activation_functions:
            raise ValueError(("The activation function '%s' of the estimator "
                              "is not supported.") % self.hidden_activation)

        # Output activation function ('softmax' or 'logistic'):
        self.output_activation = self.estimator.out_activation_
        if self.output_activation not in self.output_activation_functions:
            raise ValueError(("The activation function '%s' of the estimator "
                              "is not supported.") % self.output_activation)

        self.n_layers = self.estimator.n_layers_
        self.n_hidden_layers = self.estimator.n_layers_ - 2

        self.n_inputs = len(self.estimator.coefs_[0])
        self.n_outputs = self.estimator.n_outputs_

        self.hidden_layer_sizes = self.estimator.hidden_layer_sizes
        if isinstance(self.hidden_layer_sizes, int):
            self.hidden_layer_sizes = [self.hidden_layer_sizes]
        self.hidden_layer_sizes = list(self.hidden_layer_sizes)

        self.layer_units = \
            [self.n_inputs] + self.hidden_layer_sizes + [self.estimator.n_outputs_]

        # Weights:
        self.coefficients = self.estimator.coefs_

        # Bias:
        self.intercepts = self.estimator.intercepts_

        # Binary or multiclass classifier?
        self.is_binary = self.n_outputs == 1
        self.prefix = 'binary' if self.is_binary else 'multi'

    @property
    def hidden_activation_functions(self):
        """Get list of supported activation functions for the hidden layers."""
        return ['relu', 'identity', 'tanh', 'logistic']

    @property
    def output_activation_functions(self):
        """Get list of supported activation functions for the output layer."""
        return ['softmax', 'logistic']

    def export(self, class_name='Brain', method_name='predict', use_repr=True):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name: string, default: 'Brain'
            The name of the class in the returned result.
        :param method_name: string, default: 'predict'
            The name of the method in the returned result.
        :param use_repr : bool, default True
            Whether to use repr() for floating-point values or not.

        Returns
        -------
        :return : string
            The transpiled algorithm with the defined placeholders.
        """
        self.class_name = class_name
        self.method_name = method_name
        self.use_repr = use_repr

        if self.target_method == 'predict':
            return self.predict()

    def predict(self):
        """
        Transpile the predict method.

        Returns
        -------
        :return : string
            The transpiled predict method as string.
        """
        return self.create_class(self.create_method())

    def create_method(self):
        """
        Build the estimator method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """

        temp_arr = self.temp('arr')
        temp_arr__ = self.temp('arr[][]')
        temp_arr___ = self.temp('arr[][][]')

        # Activations:
        layers = list(self._get_activations())
        layers = ', '.join(['atts'] + layers)
        layers = temp_arr__.format(type='double', name='layers', values=layers)

        # Coefficients (weights):
        coefficients = []
        for layer in self.coefficients:
            layer_weights = []
            for weights in layer:
                weights = ', '.join([self.repr(w) for w in weights])
                layer_weights.append(temp_arr.format(weights))
            layer_weights = ', '.join(layer_weights)
            coefficients.append(temp_arr.format(layer_weights))
        coefficients = ', '.join(coefficients)
        coefficients = temp_arr___.format(type='double',
                                          name='COEFFICIENTS',
                                          values=coefficients)

        # Intercepts (biases):
        intercepts = list(self._get_intercepts())
        intercepts = ', '.join(intercepts)
        intercepts = temp_arr__.format(type='double',
                                       name='INTERCEPTS',
                                       values=intercepts)

        method_type = 'method.{}'.format(self.prefix)
        temp_method = self.temp(method_type, skipping=True, n_indents=1)
        method = temp_method.format(class_name=self.class_name,
                                    method_name=self.method_name,
                                    n_features=self.n_inputs,
                                    n_classes=self.n_outputs,
                                    layers=layers,
                                    coefficients=coefficients,
                                    intercepts=intercepts)
        n_indents = 1 if self.target_language in ['js'] else 0
        out = self.indent(method, n_indents=n_indents, skipping=True)
        return out

    def create_class(self, method):
        """
        Build the estimator class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        hidden_act_type = 'activation_fn.' + self.hidden_activation
        n_indents = 1 if self.target_language in ['java'] else 2
        hidden_act = self.temp(hidden_act_type, skipping=True,
                               n_indents=n_indents)
        output_act_type = 'output_fn.' + self.output_activation
        output_act = self.temp(output_act_type, skipping=True,
                               n_indents=n_indents)

        temp_class = self.temp('class')
        file_name = '{}.js'.format(self.class_name.lower())
        out = temp_class.format(class_name=self.class_name,
                                method_name=self.method_name, method=method,
                                n_features=self.n_inputs,
                                activation_function=hidden_act,
                                output_function=output_act,
                                file_name=file_name)
        return out

    def _get_intercepts(self):
        """
        Concatenate all intercepts of the classifier.
        """
        temp_arr = self.temp('arr')
        for layer in self.intercepts:
            inter = ', '.join([self.repr(b) for b in layer])
            yield temp_arr.format(inter)

    def _get_activations(self):
        """
        Concatenate the layers sizes of the classifier except the input layer.
        """
        temp_arr = self.temp('new_arr')
        for layer in self.layer_units[1:]:
            yield temp_arr.format(type='double',
                                  values=(str(int(layer))),
                                  fill_with='.0')
