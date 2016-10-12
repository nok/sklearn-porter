from Classifier import Classifier


class DecisionTreeClassifier(Classifier):
    """
    See also
    --------
    sklearn.tree.DecisionTreeClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    SUPPORT = {'predict': ['java', 'js']}

    # @formatter:off
    TEMPLATE = {
        'java': {
            'if':       ('{0}if (atts[{1}] <= {2}) {{'),
            'else':     ('{0}}} else {{'),
            'endif':    ('{0}}}'),
            'arr':      ('{0}classes[{1}] = {2}'),
            'join':     ('; '),
            'method': (
                'public static int {method_name}(float[] atts) {{ \n'
                '    if (atts.length != {n_features}) {{ return -1; }}; \n'
                '    int[] classes = new int[{n_classes}];'
                '    {branches} \n'
                '    int class_idx = 0; \n'
                '    int class_val = classes[0]; \n'
                '    for (int i = 1; i < {n_classes}; i++) {{ \n'
                '        if (classes[i] > class_val) {{ \n'
                '            class_idx = i; \n'
                '            class_val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}}'
            ),
            'class': (
                'class {class_name} {{ \n'
                '    {method} \n'
                '    public static void main(String[] args) {{ \n'
                '        if (args.length == {n_features}) {{ \n'
                '            float[] atts = new float[args.length]; \n'
                '            for (int i = 0, l = args.length; i < l; i++) {{ \n'
                '                atts[i] = Float.parseFloat(args[i]); \n'
                '            }} \n'
                '            System.out.println({class_name}.{method_name}(atts)); \n'
                '        }} \n'
                '    }} \n'
                '}}'
            )
        },
        'js': {
            'if':       ('{0}if (atts[{1}] <= {2}) {{'),
            'else':     ('{0}}} else {{'),
            'endif':    ('{0}}}'),
            'arr':      ('{0}classes[{1}] = {2}'),
            'join':     ('; '),
            'method': (
                'var {method_name} = function(atts) {{ \n'
                '    if (atts.length != {n_features}) {{ return -1; }}; \n'
                '    var classes = new Array({n_classes});'
                '    {branches} \n'
                '    var class_idx = 0, class_val = classes[0]; \n'
                '    for (var i = 1; i < {n_classes}; i++) {{ \n'
                '        if (classes[i] > class_val) {{ \n'
                '            class_idx = i; \n'
                '            class_val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}};'
            ),
            'class': ('{method}')
        }
    }
    # @formatter:on


    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(DecisionTreeClassifier, self).__init__(language, method_name, class_name)


    def port(self, model):
        """Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : DecisionTreeClassifier
            An instance of a trained DecisionTreeClassifier classifier.
        """
        super(self.__class__, self).port(model)
        self.n_features = model.n_features_
        self.n_classes = model.n_classes_
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


    def create_branches(self, L, R, T, value, features, node, depth):
        """Parse and port a single tree model.

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
        str = ''
        ind = '\n' + '    ' * depth
        if T[node] != -2.:
            str += self.temp('if').format(ind, features[node], repr(T[node]))
            if L[node] != -1.:
                str += self.create_branches(L, R, T, value, features, L[node], depth+1)
            str += self.temp('else').format(ind)
            if R[node] != -1.:
                str += self.create_branches(L, R, T, value, features, R[node], depth+1)
            str += self.temp('endif').format(ind)
        else:
            classes = []
            for class_idx, rate in enumerate(value[node][0]):
                classes.append(self.temp('arr').format(ind, class_idx, int(rate)))
            str += self.temp('join').join(classes) + self.temp('join')
        return str


    def create_tree(self):
        """Parse and build the tree branches.

        Returns
        -------
        :return out : string
            The tree branches as string.
        """
        feature_indices = []
        for idx in self.model.tree_.feature:
            feature_indices.append([str(jdx) for jdx in range(self.n_features)][idx])
        return self.create_branches(
            self.model.tree_.children_left,
            self.model.tree_.children_right,
            self.model.tree_.threshold,
            self.model.tree_.value,
            feature_indices, 0, 1)


    def create_method(self):
        """Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        return self.temp('method').format(
            method_name=self.method_name,
            n_features=self.n_features,
            n_classes=self.n_classes,
            branches=self.create_tree())


    def create_class(self, method):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            class_name=self.class_name,
            method_name=self.method_name,
            n_features=self.n_features,
            method=method)
