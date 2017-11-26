# -*- coding: utf-8 -*-

import os
import json
from json import encoder
from sklearn_porter.estimator.classifier.Classifier import Classifier


class DecisionTreeClassifier(Classifier):
    """
    See also
    --------
    sklearn.tree.DecisionTreeClassifier

    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    SUPPORTED_METHODS = ['predict']

    # @formatter:off
    TEMPLATES = {
        'c': {
            'if':       'if (features[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
            'arr_scope': '{{{0}}}',
            'arr[]':    '{type} {name}[{n}] = {{{values}}};',
            'arr[][]':  '{type} {name}[{n}][{m}] = {{{values}}};',
        },
        'go': {
            'if':       'if features[{0}] {1} {2} {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '\t',
            'join':     '',
            'arr_scope': '{{{0}}}',
            'arr[]':    '{name} := []{type} {{{values}}}',
            'arr[][]':  '{name} := [][]{type} {{{values}}}',
        },
        'java': {
            'if':       'if (features[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
            'type':     '{0}',
            'arr_scope': '{{{0}}}',
            'arr[]':    '{type}[] {name} = {{{values}}};',
            'arr[][]':  '{type}[][] {name} = {{{values}}};',
        },
        'js': {
            'if':       'if (features[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
            'type':     '{0}',
            'arr_scope': '[{0}]',
            'arr[]':    'var {name} = [{values}];',
            'arr[][]':  'var {name} = [{values}];',
        },
        'php': {
            'if':       'if ($features[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      '$classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
            'arr_scope': '[{0}]',
            'arr[]':    '${name} = [{values}];',
            'arr[][]':  '${name} = [{values}];',
        },
        'ruby': {
            'if':       'if features[{0}] {1} {2}',
            'else':     'else',
            'endif':    'end',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     ' ',
            'arr_scope': '[{0}]',
            'arr[]':    '{name} = [{values}]',
            'arr[][]':  '{name} = [{values}]',
        }
    }
    # @formatter:on

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param estimator : AdaBoostClassifier
            An instance of a trained DecisionTreeClassifier estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(DecisionTreeClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)
        self.estimator = estimator

    def export(self, class_name, method_name,
               export_data=False, export_dir='.',
               embed_data=False, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name: string
            The name of the class in the returned result.
        :param method_name: string
            The name of the method in the returned result.

        Returns
        -------
        :return : string
            The transpiled algorithm with the defined placeholders.
        """

        # Arguments:
        self.class_name = class_name
        self.method_name = method_name

        # Estimator:
        est = self.estimator

        self.n_features = est.n_features_
        self.n_classes = len(self.estimator.tree_.value.tolist()[0][0])

        temp_arr_scope = self.temp('arr_scope')
        temp_arr_ = self.temp('arr[]')
        temp_arr__ = self.temp('arr[][]')

        left_childs = [str(e) for e in self.estimator.tree_.children_left]
        left_childs = temp_arr_.format(type='int', name='lChilds',
                                       values=', '.join(left_childs),
                                       n=len(left_childs))
        self.left_childs = left_childs

        right_childs = [str(e) for e in self.estimator.tree_.children_right]
        right_childs = temp_arr_.format(type='int', name='rChilds',
                                        values=', '.join(right_childs),
                                        n=len(right_childs))
        self.right_childs = right_childs

        thresholds = [self.repr(e) for e in self.estimator.tree_.threshold.tolist()]
        type_ = 'float64' if self.target_language == 'go' else 'double'
        thresholds = temp_arr_.format(type=type_, name='thresholds',
                                      values=', '.join(thresholds),
                                      n=len(thresholds))
        self.thresholds = thresholds

        indices = [str(e) for e in self.estimator.tree_.feature]
        indices = temp_arr_.format(type='int', name='indices',
                                   values=', '.join(indices), n=len(indices))
        self.indices = indices

        classes = self.estimator.tree_.value.tolist()
        n = len(classes)
        m = self.n_classes
        classes = [', '.join([str(int(x)) for x in e[0]]) for e in classes]
        classes = ', '.join([temp_arr_scope.format(v) for v in classes])
        classes = temp_arr__.format(type='int', name='classes', values=classes,
                                    n=n, m=m)
        self.classes = classes

        if self.target_method == 'predict':
            # Exported:
            if export_data and os.path.isdir(export_dir):
                self.export_data(export_dir)
                return self.predict('exported')
            # Embedded:
            if embed_data:
                return self.predict('embedded')
            # Separated:
            return self.predict('separated')

    def export_data(self, export_dir):
        model_data = {
            'leftChilds': self.estimator.tree_.children_left.tolist(),
            'rightChilds': self.estimator.tree_.children_right.tolist(),
            'thresholds': self.estimator.tree_.threshold.tolist(),
            'indices': self.estimator.tree_.feature,
            'classes': self.estimator.tree_.value.tolist()
        }
        encoder.FLOAT_REPR = lambda o: self.repr(o)
        path = os.path.join(export_dir, 'data.json')
        with open(path, 'w') as fp:
            json.dump(model_data, fp)

    def predict(self, temp_type='separated'):
        """
        Transpile the predict method.

        Returns
        -------
        :return : string
            The transpiled predict method as string.
        """

        if temp_type == 'exported':
            temp = self.temp('exported.class')
            return temp.format(class_name=self.class_name,
                               method_name=self.method_name,
                               n_features=self.n_features)

        if temp_type == 'separated':
            temp = self.temp('separated.class')
            return temp.format(**self.__dict__)

        if temp_type == 'embedded':
            n_indents = 1 if self.target_language in ['java', 'js',
                                                      'php', 'ruby'] else 0
            branches = self.indent(self.create_tree(), n_indents=1)
            meth_temp = self.temp('embedded.method', n_indents=n_indents,
                                  skipping=True)
            meth = meth_temp.format(class_name=self.class_name,
                                    method_name=self.method_name,
                                    n_classes=self.n_classes,
                                    n_features=self.n_features,
                                    branches=branches)
            temp_class = self.temp('embedded.class')
            return temp_class.format(class_name=self.class_name,
                                     method_name=self.method_name,
                                     n_classes=self.n_classes,
                                     n_features=self.n_features,
                                     method=meth)

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
        out = ''
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
            temp = self.temp('arr', n_indents=depth)
            for i, rate in enumerate(value[node][0]):
                clazz = temp.format(i, int(rate))
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
        indentation = 1 if self.target_language in ['java', 'js',
                                                    'php', 'ruby'] else 0
        return self.create_branches(
            self.estimator.tree_.children_left,
            self.estimator.tree_.children_right,
            self.estimator.tree_.threshold,
            self.estimator.tree_.value,
            self.indices, 0, indentation)