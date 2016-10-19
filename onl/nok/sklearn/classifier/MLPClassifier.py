from Classifier import Classifier


class MLPClassifier(Classifier):
    """
    See also
    --------
    sklearn.neural_network.MLPClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.neural_network.MLPClassifier.html
    """

    SUPPORT = {'predict': ['java']}

    # @formatter:off
    TEMPLATE = {
        'java': {
            'type': ('{0}'),
            'arr': ('{{{0}}}'),
            'new_arr': ('new double[{0}]'),
            'arr[]': ('double[] {name} = {{{values}}};'),
            'arr[][]': ('double[][] {name} = {{{values}}};'),
            'arr[][][]': ('double[][][] {name} = {{{values}}};'),
            'method': """
    public static int {method_name}(double[] atts) {{
        if (atts.length != {n_features}) {{ return -1; }}

        {activations}
        {coefficients}
        {intercepts}

        for (int i = 0; i < activations.length - 1; i++) {{
            for (int j = 0; j < activations[i + 1].length; j++) {{
                for (int l = 0; l < activations[i].length; l++) {{
                    activations[i + 1][j] += activations[i][l] * coefficients[i][l][j];
                }}
                activations[i + 1][j] += intercepts[i][j];
                if ((i + 1) != (activations.length - 1)) {{
                    activations[i + 1] = {class_name}.hidden_activation(activations[i + 1]);
                }}
            }}
        }}
        activations[activations.length - 1] = {class_name}.output_activation(activations[activations.length - 1]);

        int class_idx = -1;
        double class_val = -1;
        for (int i = 0, l = activations[activations.length - 1].length; i < l; i++) {{
            if (activations[activations.length - 1][i] > class_val) {{
                class_val = activations[activations.length - 1][i];
                class_idx = i;
            }}
        }}
        return class_idx;
    }}
""",
            'class': """
class {class_name} {{

    public static double activation_relu(double v) {{
        return Math.max(0, v);
    }}

    public static double activation_tanh(double v) {{
        return Math.tanh(v);
    }}

    public static double activation_identity(double v) {{
        return v;
    }}

    // public static double activation_logistic(double v) {{
    //    return 1. / (1. + Math.exp(-v));
    // }}
    // public static double[] activation_logistic(double[] v) {{
    //     for (int i = 0, l = v.length; i < l; i++) {{
    //         v[i] = {class_name}.activation_logistic(v[i]);
    //     }}
    //     return v;
    // }}

    public static double[] activation_softmax(double[] v) {{
        double max = Double.NEGATIVE_INFINITY;
        for (double x : v) {{
            if (x > max) {{
                max = x;
            }}
        }}
        for (int i = 0, l = v.length; i < l; i++) {{
            v[i] = Math.exp(v[i] - max);
        }}
        double sum = 0.0;
        for (double x : v) {{
            sum += x;
        }}
        for (int i = 0, l = v.length; i < l; i++) {{
            v[i] /= sum;
        }}
        return v;
    }}

    public static double[] hidden_activation(double[] v) {{
        for (int i = 0, l = v.length; i < l; i++) {{
            v[i] = {class_name}.activation_{hidden_activation}(v[i]);
        }}
        return v;
    }}

    public static double[] output_activation(double[] v) {{
        return {class_name}.activation_{final_activation}(v);
    }}

    {method}

    public static void main(String[] args) {{
        if (args.length == {n_features}) {{
            double[] atts = new double[args.length];
            for (int i = 0, l = args.length; i < l; i++) {{
                atts[i] = Double.parseDouble(args[i]);
            }}
            System.out.println({class_name}.{method_name}(atts));
        }}
    }}
}}
"""
        }
    }
    # @formatter:on


    def __init__(
            self, language='java', method_name='predict', class_name='Tmp'):
        super(MLPClassifier, self).__init__(language, method_name, class_name)


    @property
    def hidden_activation_functions(self):
        '''Get a list of supported activation functions of the hidden layers.'''
        return ['relu', 'identity', 'tanh']  # 'logistic' failed tests


    @property
    def final_activation_functions(self):
        '''Get a list of supported activation functions of the output layer.'''
        return ['softmax']  # 'logistic' failed tests


    def port(self, model):
        """Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : MLPClassifier
            An instance of a trained MLPClassifier classifier.
        """
        super(self.__class__, self).port(model)

        # Activation function ('identity', 'logistic', 'tanh' or 'relu'):
        self.hidden_activation = self.model.activation
        if self.hidden_activation not in self.hidden_activation_functions:
            raise ValueError(('The activation function \'%s\' of the model '
                              'is not supported.') % self.hidden_activation)

        # Output activation function ('softmax' or 'logistic'):
        self.final_activation = self.model.out_activation_
        if self.final_activation not in self.final_activation_functions:
            raise ValueError(('The activation function \'%s\' of the model '
                              'is not supported.') % self.final_activation)

        self.n_layers = self.model.n_layers_
        self.n_hidden_layers = self.model.n_layers_ - 2

        self.n_inputs = len(self.model.coefs_[0])
        self.n_outputs = self.model.n_outputs_

        self.hidden_layer_sizes = self.model.hidden_layer_sizes
        if type(self.hidden_layer_sizes) is int:
            self.hidden_layer_sizes = [self.hidden_layer_sizes]
        self.hidden_layer_sizes = list(self.hidden_layer_sizes)

        self.layer_units = \
            [self.n_inputs] + self.hidden_layer_sizes + [self.model.n_outputs_]

        # Weights:
        self.coefficients = self.model.coefs_

        # Bias:
        self.intercepts = self.model.intercepts_

        if self.method_name == 'predict':
            return self.predict()


    def predict(self):
        """Port the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method.
        """
        return self.create_class(self.create_method())


    def create_method(self):
        """Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        # Activations:
        ac = [self.temp('new_arr').format((str(int(layer))))
              for layer in self.layer_units[1:]]
        ac = ', '.join(['atts'] + ac)
        ac = self.temp('arr[][]').format(name='activations', values=ac)

        # Coefficients (weights):
        coefs = []
        for layer in self.coefficients:
            layer_weights = []
            for weights in layer:
                weights = ', '.join([repr(w) for w in weights])
                layer_weights.append(self.temp('arr').format(weights))
            layer_weights = ', '.join(layer_weights)
            coefs.append(self.temp('arr').format(layer_weights))
        coefs = ', '.join(coefs)
        coefs = self.temp('arr[][][]').format(name='coefficients', values=coefs)

        # Intercepts (biases):
        inters = []
        for layer in self.intercepts:
            inter = ', '.join([repr(b) for b in layer])
            inters.append(self.temp('arr').format(inter))
        inters = ', '.join(inters)
        inters = self.temp('arr[][]').format(name='intercepts', values=inters)

        return self.temp('method').format(
            class_name=self.class_name, method_name=self.method_name,
            n_features=self.n_inputs, n_classes=self.n_outputs,
            activations=ac, coefficients=coefs, intercepts=inters)


    def create_class(self, method):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            class_name=self.class_name, method_name=self.method_name,
            method=method, n_features=self.n_inputs,
            hidden_activation=self.hidden_activation,
            final_activation=self.final_activation)
