# -*- coding: utf-8 -*-

from ...Template import Template


class DecisionTreeClassifier(Template):
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
            'if':       'if (atts[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
        },
        'java': {
            'if':       'if (atts[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
        },
        'js': {
            'if':       'if (atts[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
        },
        'php': {
            'if':       'if ($atts[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      '$classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
        }
    }
    # @formatter:on

    def __init__(self, model, target_language='java', target_method='predict', **kwargs):
        super(DecisionTreeClassifier, self).__init__(model, target_language=target_language, target_method=target_method, **kwargs)
        self.model = model
        self.n_features = model.n_features_
        self.n_classes = model.n_classes_

    def export(self, class_name, method_name):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : DecisionTreeClassifier
            An instance of a trained DecisionTreeClassifier classifier.
        """
        self.class_name = class_name
        self.method_name = method_name
        if self.target_method == 'predict':
            return self.predict(class_name, method_name)

    def predict(self, class_name, method_name):
        """
        Port the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method.
        """
        method = self.create_method(class_name, method_name)
        return self.create_class(method, class_name, method_name)

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
        out = ''
        # ind = '\n' + '    ' * depth
        if t[node] != -2.:
            out += '\n'
            out += self.temp('if', n_indents=depth).format(
                features[node], '<=', repr(t[node]))
            if l[node] != -1.:
                out += self.create_branches(
                    l, r, t, value, features, l[node], depth + 1)
            out += '\n'
            out += self.temp('else', n_indents=depth)
            if r[node] != -1.:
                out += self.create_branches(
                    l, r, t, value, features, r[node], depth + 1)
            out += '\n'
            out += self.temp('endif', n_indents=depth)
        else:
            clazzes = []
            for i, rate in enumerate(value[node][0]):
                clazz = self.temp('arr', n_indents=depth).format(i, int(rate))
                clazz = '\n' + clazz
                clazzes.append(clazz)
            out += self.temp('join').join(clazzes) + self.temp('join')
        return out

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

        indentation = 1 if self.target_language in ['java', 'js', 'php'] else 0
        return self.create_branches(
            self.model.tree_.children_left,
            self.model.tree_.children_right,
            self.model.tree_.threshold,
            self.model.tree_.value,
            feature_indices, 0, indentation)

    def create_method(self, class_name, method_name):
        """
        Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        n_indents = 1 if self.target_language in ['java', 'js', 'php'] else 0
        branches = self.indent(self.create_tree(), n_indents=1)
        return self.temp('method', n_indents=n_indents, skipping=True)\
            .format(method_name=method_name, n_features=self.n_features,
                    n_classes=self.n_classes, branches=branches)

    def create_class(self, method, class_name, method_name):
        """
        Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            class_name=class_name, method_name=method_name,
            n_features=self.n_features, method=method)
