from Classifier import Classifier


class DecisionTreeClassifier(Classifier):
    """
    See also
    --------
    sklearn.tree.DecisionTreeClassifier
    """

    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(self.__class__, self).__init__(language, method_name, class_name)
        if method_name not in ['predict']:
            msg = 'The classifier does not support the given method.'
            raise ValueError(msg)

    def port(self, model):
        """Port a trained model in the syntax of a specific programming language.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier).
        """
        super(self.__class__, self).port(model)
        self.n_features = model.n_features_
        self.n_classes = model.n_classes_
        if self.method_name == 'predict':
            return self.predict()

    def predict(self):
        """Translate the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method.
        """
        str_class = self.create_class()
        str_method = self.create_method()
        out = str_class.format(str_method)
        return out

    def create_tree(self, L, R, T, value, features, node, depth):
        """Parse and port the model tree.

        Parameters
        ----------
        :param L : object
            The left children node.
        :param R : object
            The left children node.
        :param T : object
            The decision threshold.
        :param value : object
            The label or class.
        :param features : object
            The feature values.
        :param node : int
            The current node.
        :param depth : int
            The tree depth.

        Returns
        -------
        :return : string
            The ported single tree as function or method.
        """
        out = ''
        indent = '\n' + '    ' * depth
        # @formatter:off
        if T[node] != -2.:
            template = {
                'java': ('if (atts[{0}] <= {1:.6f}f) {{'),
                'js': ('if (atts[{0}] <= {1:.6f}) {{')
            }
            out += indent + template[self.language].format(features[node], T[node])
            if L[node] != -1.:
                out += self.create_tree(L, R, T, value, features, L[node], depth + 1)
            out += indent + '} else {'
            if R[node] != -1.:
                out += self.create_tree(L, R, T, value, features, R[node], depth + 1)
            out += indent + '}'
        else:
            for idx, val in enumerate(value[node][0]):
                classes = [indent + 'classes[{0}] = {1}'.format(idx, int(val))]
                out += ';'.join(classes) + ';'
        # @formatter:on
        return out

    def create_method(self):
        """Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        feature_indices = []
        for i in self.model.tree_.feature:
            feature_indices.append([str(j) for j in range(self.n_features)][i])

        conditions = self.create_tree(
            self.model.tree_.children_left,
            self.model.tree_.children_right,
            self.model.tree_.threshold,
            self.model.tree_.value, feature_indices, 0, 1)

        template = {
            'java': (
                # @formatter:off
                'public static int {0}(float[] atts) {{ \n'
                '    if (atts.length != {1}) {{ return -1; }}; \n\n'
                '    int[] classes = new int[{2}]; \n'
                '    {3} \n\n'
                '    int idx = 0; \n'
                '    int val = classes[0]; \n'
                '    for (int i = 1; i < {2}; i++) {{ \n'
                '        if (classes[i] > val) {{ \n'
                '            idx = i; \n'
                '            val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return idx; \n'
                '}}'
                # @formatter:on
            ),
            'js': (
                # @formatter:off
                'var {0} = function(atts) {{ \n'
                '    if (atts.length != {1}) {{ return -1; }}; \n'
                '    var classes = new Array({2}); \n'
                '    {3} \n\n'
                '    var idx = 0, val = classes[0]; \n'
                '    for (var i = 1; i < {2}; i++) {{ \n'
                '        if (classes[i] > val) {{ \n'
                '            idx = i; \n'
                '            val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return idx; \n'
                '}};'
                # @formatter:on
            )
        }
        out = template[self.language].format(
            self.method_name,
            self.n_features,
            self.n_classes,
            conditions)
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
            ),
            # Just insert the single function:
            'js': ('{{0}}')
        }
        out = template[self.language].format(
            self.class_name,
            self.n_features)
        return out
