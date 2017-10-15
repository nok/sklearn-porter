# -*- coding: utf-8 -*-

from sklearn.tree.tree import DecisionTreeClassifier
from sklearn_porter.estimator.classifier.Classifier import Classifier


class RandomForestClassifier(Classifier):
    """
    See also
    --------
    sklearn.ensemble.RandomForestClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    SUPPORTED_METHODS = ['predict']
    SUPPORTED_LANGUAGES = ['c', 'java', 'js']

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
        },
        'ruby': {
            'if':       'if atts[{0}] {1} {2}',
            'else':     'else',
            'endif':    'end',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '',
        },
    }
    # @formatter:on

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param estimator : AdaBoostClassifier
            An instance of a trained RandomForestClassifier estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(RandomForestClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)
        self.estimator = estimator

        # Check type of base estimators:
        if not isinstance(estimator.base_estimator, DecisionTreeClassifier):
            msg = "The classifier doesn't support the given base estimator %s."
            raise ValueError(msg, estimator.base_estimator)

        # Check number of base estimators:
        if not estimator.n_estimators > 0:
            msg = "The classifier hasn't any base estimators."
            raise ValueError(msg)

        self.n_classes = estimator.n_classes_
        self.estimators = []
        self.n_estimators = 0
        for idx in range(self.estimator.n_estimators):
            self.estimators.append(self.estimator.estimators_[idx])
            self.n_estimators += 1
            self.n_features = self.estimator.estimators_[idx].n_features_

    def export(self, class_name, method_name, use_repr=True):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name: string
            The name of the class in the returned result.
        :param method_name: string
            The name of the method in the returned result.
        :param use_repr : bool, default True
            Whether to use repr() for floating-point values or not.

        Returns
        -------
        :return : string
            The transpiled algorithm with the defined placeholders.
        """
        self.class_name = class_name
        self.method_name = method_name
        self.use_repr = use_repr
        if self.target_method == 'predict':
            return self.predict()

    def predict(self):
        """
        Transpile the predict method.

        Returns
        -------
        :return : string
            The transpiled predict method as string.
        """
        return self.create_class(self.create_method())

    def create_branches(self, left_nodes, right_nodes, threshold,
                        value, features, node, depth):
        """
        Parse and port a single tree estimator.

        Parameters
        ----------
        :param left_nodes : object
            The left children node.
        :param right_nodes : object
            The left children node.
        :param threshold : object
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
        out = ''  # returned output
        if threshold[node] != -2.:
            out += '\n'
            temp = self.temp('if', n_indents=depth)
            out += temp.format(features[node], '<=', self.repr(threshold[node]))
            if left_nodes[node] != -1.:
                out += self.create_branches(
                    left_nodes, right_nodes, threshold, value,
                    features, left_nodes[node], depth + 1)
            out += '\n'
            out += self.temp('else', n_indents=depth)
            if right_nodes[node] != -1.:
                out += self.create_branches(
                    left_nodes, right_nodes, threshold, value,
                    features, right_nodes[node], depth + 1)
            out += '\n'
            out += self.temp('endif', n_indents=depth)
        else:
            clazzes = []
            temp_arr = self.temp('arr', n_indents=depth)
            for i, rate in enumerate(value[node][0]):
                clazz = temp_arr.format(i, int(rate))
                clazz = '\n' + clazz
                clazzes.append(clazz)
            out += self.temp('join').join(clazzes) + self.temp('join')
        return out

    def create_single_method(self, estimator_index, estimator):
        """
        Port a method for a single tree.

        Parameters
        ----------
        :param estimator_index : int
            The estimator index.
        :param estimator : RandomForestClassifier
            The estimator.

        Returns
        -------
        :return : string
            The created method.
        """
        indices = []
        for i in estimator.tree_.feature:
            n_features = estimator.n_features_
            if self.n_features > 1 or (self.n_features == 1 and i >= 0):
                indices.append([str(j) for j in range(n_features)][i])

        tree_branches = self.create_branches(
            estimator.tree_.children_left, estimator.tree_.children_right,
            estimator.tree_.threshold, estimator.tree_.value, indices, 0, 1)

        temp_single_method = self.temp('single_method')
        out = temp_single_method.format(method_name=self.method_name,
                                        method_id=str(estimator_index),
                                        n_classes=self.n_classes,
                                        tree_branches=tree_branches)
        return out

    def create_method(self):
        """
        Build the estimator methods or functions.

        Returns
        -------
        :return out : string
            The built methods as merged string.
        """
        # Generate method or function names:
        fn_names = []
        temp_method_calls = self.temp('method_calls', n_indents=2, skipping=True)
        for idx, estimator in enumerate(self.estimators):
            fn_name = self.method_name + '_' + str(idx)
            fn_name = temp_method_calls.format(class_name=self.class_name,
                                               method_name=fn_name)
            fn_names.append(fn_name)
        fn_names = '\n'.join(fn_names)
        fn_names = self.indent(fn_names, n_indents=1, skipping=True)

        # Generate related trees:
        fns = []
        for idx, estimator in enumerate(self.estimators):
            tree = self.create_single_method(idx, estimator)
            fns.append(tree)
        fns = '\n'.join(fns)

        # Merge generated content:
        n_indents = 1 if self.target_language in ['java', 'js', 'php', 'ruby'] else 0
        temp_method = self.temp('method')
        out = temp_method.format(method_name=self.method_name,
                                 method_calls=fn_names, methods=fns,
                                 n_estimators=self.n_estimators,
                                 n_classes=self.n_classes)
        out = self.indent(out, n_indents=n_indents, skipping=True)
        return out

    def create_class(self, method):
        """
        Build the estimator class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        temp_class = self.temp('class')
        out = temp_class.format(class_name=self.class_name,
                                method_name=self.method_name,
                                method=method, n_features=self.n_features)
        return out
