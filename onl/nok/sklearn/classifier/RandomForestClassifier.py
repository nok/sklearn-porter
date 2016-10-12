import sklearn

from Classifier import Classifier


class RandomForestClassifier(Classifier):
    """
    See also
    --------
    sklearn.ensemble.RandomForestClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.RandomForestClassifier.html
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
            'single_method': (
                'public static int {1}_{0}(float[] atts) {{ \n'
                '    int[] classes = new int[{2}];'
                '    {3} \n'
                '    int class_idx = 0; \n'
                '    int class_val = classes[0]; \n'
                '    for (int i = 1; i < {2}; i++) {{ \n'
                '        if (classes[i] > class_val) {{ \n'
                '            class_idx = i; \n'
                '            class_val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}}'
            ),
            'method_calls': (
                'classes[{1}.{2}(atts)]++;'
            ),
            'method': (
                '{0} \n'
                'public static int {1}(float[] atts) {{ \n'
                '    int n_classes = {3}; \n\n'
                '    int[] classes = new int[n_classes]; \n'
                '    {4} \n\n'
                '    int class_idx = 0; \n'
                '    int class_val = classes[0]; \n'
                '    for (int i = 1; i < n_classes; i++) {{ \n'
                '        if (classes[i] > class_val) {{ \n'
                '            class_idx = i; \n'
                '            class_val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}}'
            ),
            'class': (
                'class {0} {{ \n'
                '    {2} \n'
                '    public static void main(String[] args) {{ \n'
                '        if (args.length == {3}) {{ \n'
                '            float[] atts = new float[args.length]; \n'
                '            for (int i = 0, l = args.length; i < l; i++) {{ \n'
                '                atts[i] = Float.parseFloat(args[i]); \n'
                '            }} \n'
                '            System.out.println({0}.{1}(atts)); \n'
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
            'single_method': (
                'var {1}_{0} = function(atts) {{ \n'
                '    var i = 0, classes = new Array({2});'
                '    {3} \n'
                '    var class_idx = 0, class_val = classes[0]; \n'
                '    for (i = 1; i < {2}; i++) {{ \n'
                '        if (classes[i] > class_val) {{ \n'
                '            class_idx = i; \n'
                '            class_val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}};'
            ),
            'method_calls': (
                'classes[{2}(atts)]++;'
            ),
            'method': (
                '{0} \n'
                'var {1} = function(atts) {{ \n'
                '    var i = 0, n_classes = {3}; \n'
                '    var classes = new Array(n_classes); \n'
                '    for (i = 0; i < n_classes; i++) {{ \n'
                '        classes[i] = 0; \n'
                '    }} \n\n'
                '    {4} \n\n'
                '    var class_idx = 0, class_val = classes[0]; \n'
                '    for (i = 1; i < n_classes; i++) {{ \n'
                '        if (classes[i] > class_val) {{ \n'
                '            class_idx = i; \n'
                '            class_val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}}'
            ),
            'class': ('{2}')
        }
    }
    # @formatter:on


    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(RandomForestClassifier, self).__init__(language, method_name, class_name)


    def port(self, model):
        """Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : AdaBoostClassifier
            An instance of a trained AdaBoostClassifier model.
        """

        # Check type of base estimators:
        if not isinstance(model.base_estimator, sklearn.tree.tree.DecisionTreeClassifier):
            msg = "The classifier doesn't support the given base estimator %s."
            raise ValueError(msg, model.base_estimator)

        # Check number of base estimators:
        if not model.n_estimators > 0:
            msg = "The classifier hasn't any base estimators."
            raise ValueError(msg)

        self.model = model
        self.n_classes = model.n_classes_
        self.models = []
        self.n_estimators = 0
        for idx in range(self.model.n_estimators):
            self.models.append(self.model.estimators_[idx])
            self.n_estimators += 1
            self.n_features = self.model.estimators_[idx].n_features_

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
        """Port the structure of the model.

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
            The ported structure of the tree model.
        """
        str = ''
        ind = '\n' + '    ' * depth
        if T[node] != -2.:
            str += self.temp('if').format(ind, features[node], repr(T[node]))
            if L[node] != -1.:
                str += self.create_branches(L, R, T, value, features, L[node], depth + 1)
            str += self.temp('else').format(ind)
            if R[node] != -1.:
                str += self.create_branches(L, R, T, value, features, R[node], depth + 1)
            str += self.temp('endif').format(ind)
        else:
            classes = []
            for class_idx, val in enumerate(value[node][0]):
                classes.append(self.temp('arr').format(ind, class_idx, int(val)))
            str += self.temp('join').join(classes) + self.temp('join')
        return str


    def create_single_method(self, model_index, model):
        """Port a method for a single tree.

        Parameters
        ----------
        :param model_index : int
            The model index.
        :param model : DecisionTreeClassifier
            The model.

        Returns
        -------
        :return : string
            The created method.
        """
        feature_indices = []
        for idx in model.tree_.feature:
            feature_indices.append([str(jdx) for jdx in range(model.n_features_)][idx])

        tree_branches = self.create_branches(
            model.tree_.children_left,
            model.tree_.children_right,
            model.tree_.threshold,
            model.tree_.value,
            feature_indices, 0, 1)

        suffix = ("{0:0" + str(len(str(self.n_estimators - 1))) + "d}")
        model_index = suffix.format(int(model_index))

        return self.temp('single_method').format(
            model_index,
            self.method_name,
            self.n_classes,
            tree_branches)


    def create_method(self):
        """Build the model methods or functions.

        Returns
        -------
        :return out : string
            The built methods as merged string.
        """
        # Generate method or function names:
        fn_names = []
        suffix = ("_{0:0" + str(len(str(self.n_estimators - 1))) + "d}")
        for idx, model in enumerate(self.models):
            fn_name = self.method_name + suffix.format(idx)
            fn_name = self.temp('method_calls').format(idx, self.class_name, fn_name)
            fn_names.append(fn_name)

        # Generate related trees:
        fns = []
        for idx, model in enumerate(self.models):
            tree = self.create_single_method(idx, model)
            fns.append(tree)

        # Merge generated content:
        return self.temp('method').format(
            '\n'.join(fns),
            self.method_name,
            self.n_estimators,
            self.n_classes,
            '\n'.join(fn_names))


    def create_class(self, method):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            self.class_name,
            self.method_name,
            method,
            self.n_features)
