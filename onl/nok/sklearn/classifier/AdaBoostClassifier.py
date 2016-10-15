import sklearn

from Classifier import Classifier


class AdaBoostClassifier(Classifier):
    """
    See also
    --------
    sklearn.ensemble.AdaBoostClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
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
                'public static double[] {1}_{0}(float[] atts) {{ \n'
                '    double[] classes = new double[{2}];'
                '    {3} \n\n'
                '    return classes; \n'
                '}}'
            ),
            'method_calls': (
                'preds[{0}] = {1}.{2}(atts);'
            ),
            'method': (
                '{0}'
                'public static int {1}(float[] atts) {{ \n'
                '    int n_estimators = {2}; \n'
                '    int n_classes = {3}; \n\n'
                '    double[][] preds = new double[n_estimators][]; \n'
                '    {4} \n\n'
                '    int i, j; \n'
                '    double normalizer, sum; \n'
                '    for (i = 0; i < n_estimators; i++) {{ \n'
                '        normalizer = 0.; \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            normalizer += preds[i][j]; \n'
                '        }} \n'
                '        if (normalizer == 0.) {{ \n'
                '            normalizer = 1.; \n'
                '        }} \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            preds[i][j] = preds[i][j] / normalizer; \n'
                '            if (preds[i][j] < 2.2250738585072014e-308) {{ \n'
                '                preds[i][j] = 2.2250738585072014e-308; \n'
                '            }} \n'
                '            preds[i][j] = Math.log(preds[i][j]); \n'
                '        }} \n'
                '        sum = 0.; \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            sum += preds[i][j]; \n'
                '        }} \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            preds[i][j] = (n_classes - 1) * (preds[i][j] - (1. / n_classes) * sum); \n'
                '        }} \n'
                '    }} \n'
                '    double[] classes = new double[n_classes]; \n'
                '    for (i = 0; i < n_estimators; i++) {{ \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            classes[j] += preds[i][j]; \n'
                '        }} \n'
                '    }} \n'
                '    int idx = 0; \n'
                '    double val = Double.NEGATIVE_INFINITY; \n'
                '    for (i = 0; i < n_classes; i++) {{ \n'
                '        if (classes[i] > val) {{ \n'
                '            idx = i; \n'
                '            val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return idx; \n'
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
                '    var classes = new Array({2});'
                '    {3} \n\n'
                '    return classes; \n'
                '}};'
            ),
            'method_calls': (
                'preds[{0}] = {2}(atts);'
            ),
            'method': (
                '{0} \n'
                'var {1} = function(atts) {{ \n'
                '    var n_estimators = {2}, \n'
                '        preds = new Array(n_estimators), \n'
                '        n_classes = {3}, \n'
                '        classes = new Array(n_classes), \n'
                '        normalizer, sum, idx, val, \n'
                '        i, j; \n\n'
                '    {4} \n\n'
                '    for (i = 0; i < n_estimators; i++) {{ \n'
                '        normalizer = 0.; \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            normalizer += preds[i][j]; \n'
                '        }} \n'
                '        if (normalizer == 0.) {{ \n'
                '            normalizer = 1.0; \n'
                '        }} \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            preds[i][j] = preds[i][j] / normalizer; \n'
                '            if (preds[i][j] < 2.2250738585072014e-308) {{ \n'
                '                preds[i][j] = 2.2250738585072014e-308; \n'
                '            }} \n'
                '            preds[i][j] = Math.log(preds[i][j]); \n'
                '        }} \n'
                '        sum = 0.0; \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            sum += preds[i][j]; \n'
                '        }} \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            preds[i][j] = (n_classes - 1) * (preds[i][j] - (1. / n_classes) * sum); \n'
                '        }} \n'
                '    }} \n'
                '    for (i = 0; i < n_classes; i++) {{ \n'
                '        classes[i] = 0.0; \n'
                '    }} \n'
                '    for (i = 0; i < n_estimators; i++) {{ \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            classes[j] += preds[i][j]; \n'
                '        }} \n'
                '    }} \n'
                '    idx = -1; \n'
                '    val = Number.NEGATIVE_INFINITY; \n'
                '    for (i = 0; i < n_classes; i++) {{ \n'
                '        if (classes[i] > val) {{ \n'
                '            idx = i; \n'
                '            val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return idx; \n'
                '}};'
            ),
            'class': ('{2}')
        }
    }
    # @formatter:on


    def __init__(self, language='java', method_name='predict',
                 class_name='Tmp'):
        super(AdaBoostClassifier, self).__init__(language, method_name,
                                                 class_name)


    def port(self, model):
        """Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : AdaBoostClassifier
            An instance of a trained AdaBoostClassifier model.
        """

        # Check the used algorithm type:
        if model.algorithm not in ('SAMME.R'):
            msg = "The classifier doesn't support the given algorithm %s."
            raise ValueError(msg, model.algorithm)

        # Check type of base estimators:
        if not isinstance(model.base_estimator,
                          sklearn.tree.tree.DecisionTreeClassifier):
            msg = "The classifier doesn't support the given base estimator %s."
            raise ValueError(msg, model.base_estimator)

        # Check number of base estimators:
        if not model.n_estimators > 0:
            msg = "The classifier hasn't any base estimators."
            raise ValueError(msg)

        self.model = model
        self.n_classes = model.n_classes_

        self.models = []
        self.weights = []
        self.n_estimators = 0
        for idx in range(self.model.n_estimators):
            weight = self.model.estimator_weights_[idx]
            if weight > 0:
                self.models.append(self.model.estimators_[idx])
                self.weights.append(self.model.estimator_weights_[idx])
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
                str += self.create_branches(L, R, T, value, features, L[node],
                                            depth + 1)
            str += self.temp('else').format(ind)
            if R[node] != -1.:
                str += self.create_branches(L, R, T, value, features, R[node],
                                            depth + 1)
            str += self.temp('endif').format(ind)
        else:
            classes = []
            for i, val in enumerate(value[node][0]):
                classes.append(self.temp('arr').format(ind, i, repr(val)))
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
        for i in model.tree_.feature:
            n_features = model.n_features_
            feature_indices.append([str(j) for j in range(n_features)][i])

        tree_branches = self.create_branches(model.tree_.children_left,
                                             model.tree_.children_right,
                                             model.tree_.threshold,
                                             model.tree_.value,
                                             feature_indices, 0, 1)

        return self.temp('single_method').format(str(model_index),
                                                 self.method_name,
                                                 self.n_classes, tree_branches)


    def create_method(self):
        """Build the model methods or functions.

        Returns
        -------
        :return out : string
            The built methods as merged string.
        """
        # Generate method or function names:
        fn_names = []
        suffix = ("_{0:0" + str(len(str(self.n_estimators))) + "d}")
        for idx, model in enumerate(self.models):
            cl_name = self.class_name
            fn_name = self.method_name + suffix.format(idx)
            fn_name = self.temp('method_calls').format(idx, cl_name, fn_name)
            fn_names.append(fn_name)

        # Generate related trees:
        fns = []
        for idx, model in enumerate(self.models):
            tree = self.create_single_method(idx, model)
            fns.append(tree)

        fns = '\n'.join(fns)
        fn_names = '\n'.join(fn_names)

        # Merge generated content:
        return self.temp('method').format(fns, self.method_name,
                                          self.n_estimators, self.n_classes,
                                          fn_names)


    def create_class(self, method):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(self.class_name, self.method_name,
                                         method, self.n_features)
