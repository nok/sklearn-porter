# -*- coding: utf-8 -*-

import os
import json
from json import encoder
import sklearn
from sklearn_porter.estimator.classifier.Classifier import Classifier


class AdaBoostClassifier(Classifier):
    """
    See also
    --------
    sklearn.ensemble.AdaBoostClassifier

    http://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.AdaBoostClassifier.html
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
        },
        'java': {
            'if':       'if (features[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
        },
        'js': {
            'if':       'if (features[{0}] {1} {2}) {{',
            'else':     '} else {',
            'endif':    '}',
            'arr':      'classes[{0}] = {1}',
            'indent':   '    ',
            'join':     '; ',
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
        :param estimator : AdaBoostClassifier
            An instance of a trained AdaBoostClassifier estimator.
        :param target_language : string, default: 'java'
            The target programming language.
        :param target_method : string, default: 'predict'
            The target method of the estimator.
        """
        super(AdaBoostClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)

        # Check the used algorithm type:
        if estimator.algorithm != 'SAMME.R':
            msg = "The classifier doesn't support the given algorithm %s."
            raise ValueError(msg, estimator.algorithm)

        # Check type of base estimators:
        if not isinstance(
                estimator.base_estimator,
                sklearn.tree.tree.DecisionTreeClassifier):
            msg = "The classifier doesn't support the given base estimator %s."
            raise ValueError(msg, estimator.base_estimator)

        # Check number of base estimators:
        if not estimator.n_estimators > 0:
            msg = "The classifier hasn't any base estimators."
            raise ValueError(msg)

        self.estimator = estimator

    def export(self, class_name, method_name,
               export_data=False, export_dir='.', export_filename='data.json',
               export_append_checksum=False, embed_data=True, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param class_name : string
            The name of the class in the returned result.
        :param method_name: string
            The name of the method in the returned result.
        :param export_data : bool
            Whether the model data should be saved or not.
        :param export_dir : string
            The directory where the model data should be saved.
        :param export_filename : string
            The filename of the exported model data.
        :param export_append_checksum : bool
            Whether to append the checksum to the filename or not.
        :param embed_data : bool, default: True
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

        # Basic parameters:
        self.estimators = []
        for idx in range(est.n_estimators):
            if est.estimator_weights_[idx] > 0:
                self.estimators.append(est.estimators_[idx])
        self.n_classes = est.n_classes_
        self.n_features = est.estimators_[0].n_features_
        self.n_estimators = len(self.estimator)

        if self.target_method == 'predict':
            # Exported:
            if export_data and os.path.isdir(export_dir):
                self.export_data(export_dir, export_filename,
                                 export_append_checksum)
                return self.predict('exported')
            # Embedded:
            return self.predict('embedded')

    def predict(self, temp_type):
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
        # Exported:
        if temp_type == 'exported':
            temp = self.temp('exported.class')
            return temp.format(class_name=self.class_name,
                               method_name=self.method_name,
                               n_features=self.n_features)
        # Embedded:
        if temp_type == 'embedded':
            meth = self.create_embedded_meth()
            return self.create_embedded_class(meth)

    def export_data(self, directory, filename, with_md5_hash=False):
        """
        Save model data in a JSON file.

        Parameters
        ----------
        :param directory : string
            The directory.
        :param filename : string
            The filename.
        :param with_md5_hash : bool
            Whether to append the checksum to the filename or not.
        """
        model_data = []
        for est in self.estimators:
            model_data.append({
                'childrenLeft': est.tree_.children_left.tolist(),
                'childrenRight': est.tree_.children_right.tolist(),
                'thresholds': est.tree_.threshold.tolist(),
                'classes': [e[0] for e in est.tree_.value.tolist()],
                'indices': est.tree_.feature.tolist()
            })
        encoder.FLOAT_REPR = lambda o: self.repr(o)
        json_data = json.dumps(model_data, sort_keys=True)
        if with_md5_hash:
            import hashlib
            json_hash = hashlib.md5(json_data).hexdigest()
            filename = filename.split('.json')[0] + '_' + json_hash + '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as file_:
            file_.write(json_data)

    def create_branches(self, left_nodes, right_nodes, threshold,
                        value, features, node, depth, init=False):
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
        :param init : bool, default: True
            Whether it's the initial tree or not.

        Returns
        -------
        :return out : string
            The ported single tree as function or method.
        """
        out = ''
        if threshold[node] != -2.:
            if not init:
                out += '\n'
            temp = self.temp('if', n_indents=depth, skipping=init)
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
            for i, val in enumerate(value[node][0]):
                clazz = temp.format(i, self.repr(val))
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
        :param estimator : AdaBoostClassifier
            The estimator itself.

        Returns
        -------
        :return : string
            The created method as string.
        """
        feature_indices = []
        for i in estimator.tree_.feature:
            n_features = estimator.n_features_
            if self.n_features > 1 or (self.n_features == 1 and i >= 0):
                feature_indices.append([str(j) for j in range(n_features)][i])

        tree_branches = self.create_branches(
            estimator.tree_.children_left, estimator.tree_.children_right,
            estimator.tree_.threshold, estimator.tree_.value,
            feature_indices, 0, 1, init=True)

        temp_tree = self.temp('embedded.tree')
        return temp_tree.format(method_name=self.method_name,
                                method_index=str(estimator_index),
                                methods=tree_branches,
                                n_classes=self.n_classes)

    def create_embedded_meth(self):
        """
        Build the estimator methods or functions.

        Returns
        -------
        :return : string
            The built methods as merged string.
        """
        # Generate related trees:
        fns = []
        for idx, estimator in enumerate(self.estimators):
            tree = self.create_single_method(idx, estimator)
            fns.append(tree)
        fns = '\n'.join(fns)

        # Generate method or function names:
        fn_names = ''
        if self.target_language in ['c', 'java']:
            fn_names = []
            temp_method_calls = self.temp('embedded.method_calls', n_indents=2,
                                          skipping=True)
            for idx, estimator in enumerate(self.estimators):
                cl_name = self.class_name
                fn_name = self.method_name + '_' + str(idx)
                fn_name = temp_method_calls.format(class_name=cl_name,
                                                   method_name=fn_name,
                                                   method_index=idx)
                fn_names.append(fn_name)
            fn_names = '\n'.join(fn_names)
            fn_names = self.indent(fn_names, n_indents=1, skipping=True)

        # Merge generated content:
        n_indents = 1 if self.target_language in ['java', 'js'] else 0
        temp_method = self.temp('embedded.method')
        method = temp_method.format(method_name=self.method_name,
                                    class_name=self.class_name,
                                    method_calls=fn_names, methods=fns,
                                    n_estimators=self.n_estimators,
                                    n_classes=self.n_classes)
        return self.indent(method, n_indents=n_indents, skipping=True)

    def create_embedded_class(self, method):
        """
        Build the estimator class.

        Returns
        -------
        :return : string
            The built class as string.
        """
        temp_class = self.temp('embedded.class')
        return temp_class.format(class_name=self.class_name,
                                 method_name=self.method_name, method=method,
                                 n_features=self.n_features)
