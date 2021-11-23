# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

from sklearn_porter.estimator.classifier.Classifier import Classifier


class DecisionTreeClassifier(Classifier):
    """
    See also
    --------
    sklearn.tree.DecisionTreeClassifier

    http://scikit-learn.org/stable/modules/generated/
    sklearn.tree.DecisionTreeClassifier.html
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
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param estimator : DecisionTreeClassifier
            An instance of a trained DecisionTreeClassifier estimator.
        :param target_language : string, default: 'java'
            The target programming language.
        :param target_method : string, default: 'predict'
            The target method of the estimator.
        """
        super(DecisionTreeClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)
        self.estimator = estimator

    def export(self, class_name, method_name, export_data=False,
               export_dir='.', export_filename='data.json',
               export_append_checksum=False, embed_data=False, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param class_name : string
            The name of the class in the returned result.
        :param method_name : string
            The name of the method in the returned result.
        :param export_data : bool, default: False
            Whether the model data should be saved or not.
        :param export_dir : string, default: '.' (current directory)
            The directory where the model data should be saved.
        :param export_filename : string, default: 'data.json'
            The filename of the exported model data.
        :param export_append_checksum : bool, default: False
            Whether to append the checksum to the filename or not.
        :param embed_data : bool, default: False
            Whether the model data should be embedded in the template or not.

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

        thresholds = [self.repr(e) for e in
                      self.estimator.tree_.threshold.tolist()]
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
        
        # transfer dt to c language and is multilabel task
        if self.target_language in ['c'] and self.estimator.tree_.value.ndim == 3:
            import numpy as np
            classes = np.argmax(self.estimator.tree_.value, axis=2).tolist()
            n = len(classes)
            m = len(classes[0])
            classes = [', '.join([str(int(x)) for x in e]) for e in classes]
            classes = ', '.join([temp_arr_scope.format(v) for v in classes])
            classes = temp_arr__.format(type='int', name='classes', values=classes,
                                    n=n, m=m)
            self.n_outputs_ = self.estimator.n_outputs_

        self.classes = classes

        if self.target_method == 'predict':
            # Exported:
            if export_data and os.path.isdir(export_dir):
                self.export_data(export_dir, export_filename,
                                 export_append_checksum)
                return self.predict('exported')
            # Embedded:
            if embed_data:
                return self.predict('embedded')
            # Separated:
            return self.predict('separated')

    def export_data(self, directory, filename, with_md5_hash=False):
        """
        Save model data in a JSON file.

        Parameters
        ----------
        :param directory : string
            The directory.
        :param filename : string
            The filename.
        :param with_md5_hash : bool, default: False
            Whether to append the checksum to the filename or not.
        """
        model_data = {
            'leftChilds': self.estimator.tree_.children_left.tolist(),
            'rightChilds': self.estimator.tree_.children_right.tolist(),
            'thresholds': self.estimator.tree_.threshold.tolist(),
            'indices': self.estimator.tree_.feature.tolist(),
            'classes': [c[0] for c in self.estimator.tree_.value.tolist()]
        }
        encoder.FLOAT_REPR = lambda o: self.repr(o)
        json_data = dumps(model_data, sort_keys=True)
        if with_md5_hash:
            import hashlib
            json_hash = hashlib.md5(json_data).hexdigest()
            filename = filename.split('.json')[0] + '_' + json_hash + '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as fp:
            fp.write(json_data)

    def predict(self, temp_type='separated'):
        """
        Transpile the predict method.

        Parameters
        ----------
        :param temp_type : string
            The kind of export type (embedded, separated, exported).

        Returns
        -------
        :return : string
            The transpiled predict method as string.
        """
        if temp_type == 'exported':
            exported_temp = self.temp('exported.class')
            return exported_temp.format(class_name=self.class_name,
                                        method_name=self.method_name,
                                        n_features=self.n_features)

        if temp_type == 'separated':
            separated_temp = self.temp('separated.class')
            # transfer dt to c language and is multilabel task
            if self.target_language in ['c'] and self.estimator.tree_.value.ndim == 3:
                separated_temp = self.temp('separated.multilabels.class')
            return separated_temp.format(**self.__dict__)

        if temp_type == 'embedded':
            n_indents = 1 if self.target_language in ['java', 'js',
                                                      'php', 'ruby'] else 0
            branches = self.indent(self.create_tree(), n_indents=1)
            embedded_temp = self.temp('embedded.method', n_indents=n_indents,
                                      skipping=True)
            method = embedded_temp.format(class_name=self.class_name,
                                          method_name=self.method_name,
                                          n_classes=self.n_classes,
                                          n_features=self.n_features,
                                          branches=branches)
            embedded_temp = self.temp('embedded.class')
            return embedded_temp.format(class_name=self.class_name,
                                        method_name=self.method_name,
                                        n_classes=self.n_classes,
                                        n_features=self.n_features,
                                        method=method)

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
        :return out : string
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
        :return : string
            The tree branches as string.
        """
        feature_indices = []
        for i in self.estimator.tree_.feature:
            n_features = self.n_features
            if self.n_features > 1 or (self.n_features == 1 and i >= 0):
                feature_indices.append([str(j) for j in range(n_features)][i])
        indentation = 1 if self.target_language in ['java', 'js',
                                                    'php', 'ruby'] else 0
        return self.create_branches(
            self.estimator.tree_.children_left,
            self.estimator.tree_.children_right,
            self.estimator.tree_.threshold,
            self.estimator.tree_.value,
            feature_indices, 0, indentation)
