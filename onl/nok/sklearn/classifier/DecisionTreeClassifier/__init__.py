from .. import Classifier


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
        },
        'js': {
            'if':       ('{0}if (atts[{1}] <= {2}) {{'),
            'else':     ('{0}}} else {{'),
            'endif':    ('{0}}}'),
            'arr':      ('{0}classes[{1}] = {2}'),
            'join':     ('; '),
            'class':    ('{method}')
        }
    }
    # @formatter:on


    def __init__(
            self, language='java', method_name='predict', class_name='Tmp'):
        super(DecisionTreeClassifier, self).__init__(
            language, method_name, class_name)


    def port(self, model):
        """
        Port a trained model to the syntax of a chosen programming language.

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
        """
        Port the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method.
        """
        return self.create_class(self.create_method())


    def create_branches(self, L, R, T, value, features, node, depth):
        """
        Parse and port a single tree model.

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
                str += self.create_branches(
                    L, R, T, value, features, L[node], depth + 1)
            str += self.temp('else').format(ind)
            if R[node] != -1.:
                str += self.create_branches(
                    L, R, T, value, features, R[node], depth + 1)
            str += self.temp('endif').format(ind)
        else:
            classes = []
            for i, rate in enumerate(value[node][0]):
                classes.append(self.temp('arr').format(ind, i, int(rate)))
            str += self.temp('join').join(classes) + self.temp('join')
        return str


    def create_tree(self):
        """
        Parse and build the tree branches.

        Returns
        -------
        :return out : string
            The tree branches as string.
        """
        feature_indices = []
        for i in self.model.tree_.feature:
            feature_indices.append([str(j) for j in range(self.n_features)][i])
        return self.create_branches(
            self.model.tree_.children_left,
            self.model.tree_.children_right,
            self.model.tree_.threshold,
            self.model.tree_.value,
            feature_indices, 0, 1)


    def create_method(self):
        """
        Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        branches = self.indent(self.create_tree(), indentation=4)
        return self.temp('method', indentation=4).format(
            method_name=self.method_name, n_features=self.n_features,
            n_classes=self.n_classes, branches=branches)


    def create_class(self, method):
        """
        Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            class_name=self.class_name, method_name=self.method_name,
            n_features=self.n_features, method=method)
