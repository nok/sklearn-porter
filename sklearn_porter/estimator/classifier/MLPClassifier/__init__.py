# -*- coding: utf-8 -*-

import os
import json
from json import encoder
from sklearn_porter.estimator.classifier.Classifier import Classifier


class MLPClassifier(Classifier):
    """
    See also
    --------
    sklearn.neural_network.MLPClassifier

    http://scikit-learn.org/stable/modules/generated/
    sklearn.neural_network.MLPClassifier.html
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
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param estimator : MLPClassifier
            An instance of a trained MLPClassifier estimator.
        :param target_language : string, default: 'java'
            The target programming language.
        :param target_method : string, default: 'predict'
            The target method of the estimator.
        """
        super(MLPClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)

        # Activation function ('identity', 'logistic', 'tanh' or 'relu'):
        hidden_activation = estimator.activation
        if hidden_activation not in self.hidden_activation_functions:
            raise ValueError(("The activation function '%s' of the estimator "
                              "is not supported.") % hidden_activation)

        # Output activation function ('softmax' or 'logistic'):
        output_activation = estimator.out_activation_
        if output_activation not in self.output_activation_functions:
            raise ValueError(("The activation function '%s' of the estimator "
                              "is not supported.") % output_activation)

        self.estimator = estimator

    @property
    def hidden_activation_functions(self):
        """Get list of supported activation functions for the hidden layers."""
        return ['relu', 'identity', 'tanh', 'logistic']

    @property
    def output_activation_functions(self):
        """Get list of supported activation functions for the output layer."""
        return ['softmax', 'logistic']

    def export(self, class_name, method_name, export_data=False,
               export_dir='.', export_filename='data.json',
               export_append_checksum=False, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name : string
            The name of the class in the returned result.
        :param method_name : string
            The name of the method in the returned result.
        :param export_data : bool, default: False
            Whether the model data should be saved or not.
        :param export_dir : string, default: '.' (current directory)
            The directory where the model data should be saved.
        :param export_filename : string, default: 'data.json'
            The filename of the exported model data.
        :param export_append_checksum : bool, default: False
            Whether to append the checksum to the filename or not.

        Returns
        -------
        :return : string
            The transpiled algorithm with the defined placeholders.
        """
        # Arguments:
        self.class_name = class_name
        self.method_name = method_name

        # Estimator:
        est = self.estimator

        self.output_activation = est.out_activation_
        self.hidden_activation = est.activation

        self.n_layers = est.n_layers_
        self.n_hidden_layers = est.n_layers_ - 2

        self.n_inputs = len(est.coefs_[0])
        self.n_outputs = est.n_outputs_

        self.hidden_layer_sizes = est.hidden_layer_sizes
        if isinstance(self.hidden_layer_sizes, int):
            self.hidden_layer_sizes = [self.hidden_layer_sizes]
        self.hidden_layer_sizes = list(self.hidden_layer_sizes)

        self.layer_units = \
            [self.n_inputs] + self.hidden_layer_sizes + [est.n_outputs_]

        # Weights:
        self.coefficients = est.coefs_

        # Bias:
        self.intercepts = est.intercepts_

        # Binary or multiclass classifier?
        self.is_binary = self.n_outputs == 1
        self.prefix = 'binary' if self.is_binary else 'multi'

        if self.target_method == 'predict':
            # Exported:
            if export_data and os.path.isdir(export_dir):
                self.export_data(export_dir, export_filename,
                                 export_append_checksum)
                return self.predict('exported')
            # Separated:
            return self.predict('separated')

    def predict(self, temp_type):
        """
        Transpile the predict method.

        Parameters
        ----------
        :param temp_type : string
            The kind of export type (embedded, separated, exported).

        Returns
        -------
        :return : string
            The transpiled predict method as string.
        """
        # Exported:
        if temp_type == 'exported':
            temp = self.temp('exported.class')
            return temp.format(class_name=self.class_name,
                               method_name=self.method_name)
        # Separated:
        temp_arr = self.temp('arr')
        temp_arr_ = self.temp('arr[]')
        temp_arr__ = self.temp('arr[][]')
        temp_arr___ = self.temp('arr[][][]')

        # Activations:
        layers = list(self._get_activations())
        layers = ', '.join(layers)
        layers = temp_arr_.format(type='int', name='layers', values=layers)

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
                                          name='weights',
                                          values=coefficients)

        # Intercepts (biases):
        intercepts = list(self._get_intercepts())
        intercepts = ', '.join(intercepts)
        intercepts = temp_arr__.format(type='double',
                                       name='bias',
                                       values=intercepts)

        temp_class = self.temp('separated.class')
        file_name = '{}.js'.format(self.class_name.lower())
        return temp_class.format(class_name=self.class_name,
                                 method_name=self.method_name,
                                 hidden_activation=self.hidden_activation,
                                 output_activation=self.output_activation,
                                 n_features=self.n_inputs,
                                 weights=coefficients,
                                 bias=intercepts,
                                 layers=layers,
                                 file_name=file_name)

    def export_data(self, directory, filename, with_md5_hash=False):
        """
        Save model data in a JSON file.

        Parameters
        ----------
        :param directory : string
            The directory.
        :param filename : string
            The filename.
        :param with_md5_hash : bool, default: False
            Whether to append the checksum to the filename or not.
        """
        model_data = {
            'layers': [int(l) for l in list(self._get_activations())],
            'weights': [c.tolist() for c in self.coefficients],
            'bias': [i.tolist() for i in self.intercepts],
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation
        }
        encoder.FLOAT_REPR = lambda o: self.repr(o)
        json_data = json.dumps(model_data, sort_keys=True)
        if with_md5_hash:
            import hashlib
            json_hash = hashlib.md5(json_data).hexdigest()
            filename = filename.split('.json')[0] + '_' + json_hash + '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as fp:
            fp.write(json_data)

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
        return [str(x) for x in self.layer_units[1:]]
