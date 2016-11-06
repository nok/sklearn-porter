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
            'if':       ('\nif (atts[{0}] {1} {2}) {{'),
            'else':     ('\n} else {'),
            'endif':    ('\n}'),
            'arr':      ('\nclasses[{0}] = {1}\n'),
            'indent':   ('    '),
            'join':     ('; '),
        },
        'js': {
            'if':       ('\nif (atts[{0}] {1} {2}) {{'),
            'else':     ('\n} else {'),
            'endif':    ('\n}'),
            'arr':      ('\nclasses[{0}] = {1}\n'),
            'indent':   ('    '),
            'join':     ('; '),
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


    def create_branches(self, l, r, t, value, features, node, depth):
        """
        Parse and port a single tree model.

        Parameters
        ----------
        :param l : object
            The left children node.
        :param r : object
            The left children node.
        :param t : object
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
        # ind = '\n' + '    ' * depth
        if t[node] != -2.:
            str += self.temp('if', indentation=depth).format(
                features[node], '<=', repr(t[node]))
            if l[node] != -1.:
                str += self.create_branches(
                    l, r, t, value, features, l[node], depth + 1)
            str += self.temp('else', indentation=depth)
            if r[node] != -1.:
                str += self.create_branches(
                    l, r, t, value, features, r[node], depth + 1)
            str += self.temp('endif', indentation=depth)
        else:
            clazzes = []
            for i, rate in enumerate(value[node][0]):
                clazz = self.temp('arr', indentation=depth).format(i, int(rate))
                clazzes.append(clazz)
            str += self.temp('join').join(clazzes) + self.temp('join')
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
        branches = self.indent(self.create_tree(), indentation=1)
        return self.temp('method', indentation=1, skipping=True).format(
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
