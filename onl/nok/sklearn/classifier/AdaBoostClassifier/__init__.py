import sklearn

from ..Classifier import Classifier


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
        },
        'js': {
            'if':       ('{0}if (atts[{1}] <= {2}) {{'),
            'else':     ('{0}}} else {{'),
            'endif':    ('{0}}}'),
            'arr':      ('{0}classes[{1}] = {2}'),
            'join':     ('; '),
            'class':    ('{2}')
        }
    }
    # @formatter:on


    def __init__(self, language='java', method_name='predict',
                 class_name='Tmp'):
        super(AdaBoostClassifier, self).__init__(
            language, method_name, class_name)


    def port(self, model):
        """
        Port a trained model to the syntax of a chosen programming language.

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
        if not isinstance(
                model.base_estimator,
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
        """
        Port the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method as string.
        """
        return self.create_class(self.create_method())


    def create_branches(self, L, R, T, value, features, node, depth):
        """
        Port the structure of the model.

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
                str += self.create_branches(
                    L, R, T, value, features, L[node], depth + 1)
            str += self.temp('else').format(ind)
            if R[node] != -1.:
                str += self.create_branches(
                    L, R, T, value, features, R[node], depth + 1)
            str += self.temp('endif').format(ind)
        else:
            classes = []
            for i, val in enumerate(value[node][0]):
                classes.append(self.temp('arr').format(ind, i, repr(val)))
            str += self.temp('join').join(classes) + self.temp('join')
        return str


    def create_single_method(self, model_index, model):
        """
        Port a method for a single tree.

        Parameters
        ----------
        :param model_index : int
            The model index.
        :param model : AdaBoostClassifier
            The model itself.

        Returns
        -------
        :return : string
            The created method as string.
        """
        feature_indices = []
        for i in model.tree_.feature:
            n_features = model.n_features_
            feature_indices.append([str(j) for j in range(n_features)][i])

        tree_branches = self.create_branches(
            model.tree_.children_left, model.tree_.children_right,
            model.tree_.threshold, model.tree_.value,
            feature_indices, 0, 1)

        return self.temp('single_method').format(
            str(model_index), self.method_name,
            self.n_classes, tree_branches)


    def create_method(self):
        """
        Build the model methods or functions.

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
            fn_name = self.temp('method_calls', indentation=4).format(
                idx, cl_name, fn_name)
            fn_names.append(fn_name)

        # Generate related trees:
        fns = []
        for idx, model in enumerate(self.models):
            tree = self.create_single_method(idx, model)
            # tree = self.indent(tree, indentation=4)
            fns.append(tree)

        fns = '\n'.join(fns)
        fn_names = '\n'.join(fn_names)

        # Merge generated content:
        method = self.temp('method').format(
            fns, self.method_name, self.n_estimators, self.n_classes, fn_names)
        method = self.indent(method, indentation=4)
        return method


    def create_class(self, method):
        """
        Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            self.class_name, self.method_name, method, self.n_features)
