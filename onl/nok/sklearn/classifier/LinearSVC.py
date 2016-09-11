from Classifier import Classifier


class LinearSVC(Classifier):
    """
    See also
    --------
    sklearn.svm.LinearSVC

    http://scikit-learn.org/0.17/modules/generated/sklearn.svm.LinearSVC.html
    """

    SUPPORT = {
        'predict': ['c', 'java', 'js']
    }

    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(self.__class__, self).__init__(language, method_name, class_name)


    def port(self, model):
        """Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : LinearSVC
            An instance of a trained LinearSVC classifier.
        """
        super(self.__class__, self).port(model)

        self.n_features = len(self.model.coef_[0])
        self.n_classes = len(self.model.classes_)

        if self.method_name == 'predict':
            return self.predict()


    def predict(self):
        """Port the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method.
        """
        str_class = self.create_class()
        str_method = self.create_method()
        out = str_class.format(str_method)
        return out


    def create_method(self):
        """Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        # Coefficients:
        coefs = []
        for idx, coef in enumerate(self.model.coef_):
            template = {'c': ('{0}'), 'java': ('{0}'), 'js': ('{0}')}
            _coefs = [template[self.language].format(repr(c)) for c in coef]
            template = {'c': ('{{{0}}}'), 'java': ('{{{0}}}'), 'js': ('[{0}]')}
            _coefs = template[self.language].format(', '.join(_coefs))
            coefs.append(_coefs)
        # @formatter:off
        template = {
            'c': ('double coefs[{1}][{2}] = {{{0}}};'),
            'java': ('double[][] coefs = {{{0}}};'),
            'js': ('var coefs = [{0}];')
        }
        # @formatter:on
        coefs = template[self.language].format(
            ', '.join(coefs),
            self.n_classes,
            self.n_features)

        # Interceptions:
        template = {'c': ('{0}'), 'java': ('{0}'), 'js': ('{0}')}
        inters = [template[self.language].format(repr(i)) for i in self.model.intercept_]
        # @formatter:off
        template = {
            'c': ('double inters[{1}] = {{{0}}};'),
            'java': ('double[] inters = {{{0}}};'),
            'js': ('var inters = [{0}];')
        }
        # @formatter:on
        inters = template[self.language].format(
            ', '.join(inters),
            self.n_classes)

        # Prediction method / function:
        # @formatter:off
        template = {
            'c': (
                'int predict (float atts[4]) {{ \n'
                '    {3} \n'
                '    {4} \n'
                '    double class_val = -INFINITY; \n'
                '    int class_idx = -1; \n'
                '    for (int i = 0; i < {2}; i++) {{ \n'
                '        double prob = 0.; \n'
                '        for (int j = 0; j < {1}; j++) {{ \n'
                '            prob += coefs[i][j] * atts[j]; \n'
                '        }} \n'
                '        if (prob + inters[i] > class_val) {{ \n'
                '            class_val = prob + inters[i]; \n'
                '            class_idx = i; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}} \n'
            ),
            'java': (
                'public static int {0}(float[] atts) {{ \n'
                '    if (atts.length != {1}) {{ return -1; }} \n'
                '    {3} \n'
                '    {4} \n'
                '    int class_idx = -1; \n'
                '    double class_val = Double.NEGATIVE_INFINITY; \n'
                '    for (int i = 0; i < {2}; i++) {{ \n'
                '        double prob = 0.; \n'
                '        for (int j = 0; j < {1}; j++) {{ \n'
                '            prob += coefs[i][j] * atts[j]; \n'
                '        }} \n'
                '        if (prob + inters[i] > class_val) {{ \n'
                '            class_val = prob + inters[i]; \n'
                '            class_idx = i; \n'
                '        }}'
                '    }} \n'
                '    return class_idx; \n'
                '}}'
            ),
            'js': (
                'var {0} = function(atts) {{ \n'
                '    if (atts.length != {1}) {{ return -1; }}; \n'
                '    {3} \n'
                '    {4} \n'
                '    var class_idx = -1; \n'
                '    var class_val = Number.NEGATIVE_INFINITY; \n'
                '    for (var i = 0; i < {2}; i++) {{ \n'
                '        var prob = 0.; \n'
                '        for (var j = 0; j < {1}; j++) {{ \n'
                '            prob += coefs[i][j] * atts[j]; \n'
                '        }} \n'
                '        if (prob + inters[i] > class_val) {{ \n'
                '            class_val = prob + inters[i]; \n'
                '            class_idx = i; \n'
                '        }}'
                '    }} \n'
                '    return class_idx; \n'
                '}};'
            )
        }
        # @formatter:on

        out = template[self.language].format(
            self.method_name,
            self.n_features,
            self.n_classes,
            coefs,
            inters)

        return out


    def create_class(self):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        # @formatter:off
        template = {
            'c': (
                '#include <stdio.h> \n'
                '#include <math.h> \n\n'
                '{{0}} \n'
                'int main(int argc, const char * argv[]) {{{{ \n'
                '    float test[{1}] = {{{{1.f, 2.f, 3.f, 4.f}}}}; \n'
                '    printf("Result: %d\\n", predict(test)); \n'
                '    return 0; \n'
                '}}}}'
            ),
            'java': (
                'class {0} {{{{ \n'
                '    {{0}} \n'
                '    public static void main(String[] args) {{{{ \n'
                '        if (args.length == {1}) {{{{ \n'
                '            float[] atts = new float[args.length]; \n'
                '            for (int i = 0, l = args.length; i < l; i++) {{{{ \n'
                '                atts[i] = Float.parseFloat(args[i]); \n'
                '            }}}} \n'
                '            System.out.println({0}.predict(atts)); \n'
                '        }}}} \n'
                '    }}}} \n'
                '}}}}'
            ),
            'js': ('{{0}}')
        }
        # @formatter:on

        out = template[self.language].format(
            self.class_name,
            self.n_features)

        return out
