# -*- coding: utf-8 -*-

from ...Model import Model


class DecisionTreeClassifier(Model):
    """
    See also
    --------
    sklearn.tree.DecisionTreeClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    SUPPORTED_METHODS = ['predict']

    # @formatter:off
    TEMPLATES = {
        'c': {
            'if':       '\nif (atts[{0}] {1} {2}) {{',
            'else':     '\n} else {',
            'endif':    '\n}',
            'arr':      '\nclasses[{0}] = {1}\n',
            'indent':   '    ',
            'join':     '; ',
        },
        'java': {
            'if':       '\nif (atts[{0}] {1} {2}) {{',
            'else':     '\n} else {',
            'endif':    '\n}',
            'arr':      '\nclasses[{0}] = {1}\n',
            'indent':   '    ',
            'join':     '; ',
        },
        'js': {
            'if':       '\nif (atts[{0}] {1} {2}) {{',
            'else':     '\n} else {',
            'endif':    '\n}',
            'arr':      '\nclasses[{0}] = {1}\n',
            'indent':   '    ',
            'join':     '; ',
        },
        'php': {
            'if':       '\nif ($atts[{0}] {1} {2}) {{',
            'else':     '\n} else {',
            'endif':    '\n}',
            'arr':      '\n$classes[{0}] = {1}\n',
            'indent':   '    ',
            'join':     '; ',
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
            str += self.temp('if', n_indents=depth).format(
                features[node], '<=', repr(t[node]))
            if l[node] != -1.:
                str += self.create_branches(
                    l, r, t, value, features, l[node], depth + 1)
            str += self.temp('else', n_indents=depth)
            if r[node] != -1.:
                str += self.create_branches(
                    l, r, t, value, features, r[node], depth + 1)
            str += self.temp('endif', n_indents=depth)
        else:
            clazzes = []
            for i, rate in enumerate(value[node][0]):
                clazz = self.temp('arr', n_indents=depth).format(i, int(rate))
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
            n_features = self.n_features
            if self.n_features > 1 or (self.n_features == 1 and i >= 0):
                feature_indices.append([str(j) for j in range(n_features)][i])

        indentation = 1 if self.language in ['java', 'js', 'php'] else 0
        return self.create_branches(
            self.model.tree_.children_left,
            self.model.tree_.children_right,
            self.model.tree_.threshold,
            self.model.tree_.value,
            feature_indices, 0, indentation)

    def create_method(self):
        """
        Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        indentation = 1 if self.language in ['java', 'js', 'php'] else 0
        branches = self.indent(self.create_tree(), n_indents=1)
        return self.temp('method', n_indents=indentation, skipping=True)\
            .format(method_name=self.method_name, n_features=self.n_features,
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
