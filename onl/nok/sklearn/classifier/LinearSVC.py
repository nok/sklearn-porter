from Classifier import Classifier


class LinearSVC(Classifier):
    """
    See also
    --------
    sklearn.svm.LinearSVC

    http://scikit-learn.org/0.17/modules/generated/sklearn.svm.LinearSVC.html
    """

    SUPPORT = {
        'predict': ['java']
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
        coefs, inters = '', ''

        if self.language is 'java':

            # Coefficients:
            coefs = []
            for idx, coef in enumerate(self.model.coef_):
                _coefs = ['{0}f'.format(str(c)) for c in coef]
                _coefs = '{{ {0} }}'.format(', '.join(_coefs))
                coefs.append(_coefs)
            coefs = 'float[][] coefs = {{ {0} }};'.format(', '.join(coefs))

            # Interceptions:
            inters = ['{0}f'.format(str(interc)) for interc in self.model.intercept_]
            inters = 'float[] inters = {{ {0} }};'.format(', '.join(inters))

        template = {
            'java': (
                # @formatter:off
                'public static int {0}(float[] atts) {{ \n'
                '    if (atts.length != {1}) {{ return -1; }} \n'
                '    {3} \n'
                '    {4} \n'
                '    float[] classes = new float[{2}]; \n'
                '    for (int i = 0; i < {2}; i++) {{ \n'
                '        float prob = 0.0f; \n'
                '        for (int j = 0; j < {1}; j++) {{ \n'
                '            prob += coefs[i][j] * atts[j]; \n'
                '        }} \n'
                '        classes[i] = prob + inters[i]; \n'
                '    }} \n'
                '    int idx = 0; \n'
                '    float val = classes[0]; \n'
                '    for (int i = 1; i < {2}; i++) {{ \n'
                '        if (classes[i] > val) {{ \n'
                '            idx = i; \n'
                '            val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return idx; \n'
                '}}'
                # @formatter:on
            )
        }

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
        template = {
            'java': (
                # @formatter:off
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
                # @formatter:on
            )
        }

        out = template[self.language].format(
            self.class_name,
            self.n_features)

        return out
