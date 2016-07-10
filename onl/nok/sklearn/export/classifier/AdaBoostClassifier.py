import numpy as np
import sklearn

from onl.nok.sklearn.export.classifier.Classifier import Classifier
from onl.nok.sklearn.export.classifier.DecisionTreeClassifier import DecisionTreeClassifier


class AdaBoostClassifier(Classifier):


    @staticmethod
    def get_supported_methods():
        return [
            'predict'
        ]


    @staticmethod
    def is_supported_method(method_name):
        support = method_name in AdaBoostClassifier.get_supported_methods()
        if not support:
            raise ValueError('The classifier does not support the given method.')
        return support


    @staticmethod
    def export(model, method_name='predict', class_name="Tmp"):
        print("Ada.export()")
        if AdaBoostClassifier.is_supported_method(method_name):
            if method_name == 'predict':
                return AdaBoostClassifier.predict(model, class_name=class_name)
        return False


    @staticmethod
    def predict(model, class_name='Tmp'):
        print("Ada.predict()")
        method_name = 'predict'

        if model.algorithm not in ('SAMME', 'SAMME.R'):
            # raise ValueError("algorithm %s is not supported" % model.algorithm)
            return False

        # Check type of base estimators:
        if not isinstance(model.base_estimator, sklearn.tree.tree.DecisionTreeClassifier):
            return False

        # Check number of base estimators:
        if not model.n_estimators > 0:
            return False

        # TODO: Use local variables
        # n_features = model.estimators_[0].n_features_
        # n_classes = model.n_classes_

        function_names = []
        functions = []
        n_estimators = 0
        for idx in range(model.n_estimators):
            weight = float(model.estimator_weights_[idx])
            if weight > 0:
                n_estimators += 1
                id = ("{0:0" + str(len(str(model.n_estimators))) + "d}").format(idx)
                function_name_with_id = method_name + "_" + id
                function_names.append(function_name_with_id)
                functions.append(AdaBoostClassifier._create_trees(
                    model.estimators_[idx], function_name_with_id))

        str_tree_calls = "\n".join(['preds[{0}] = {1}.{2}(atts);'.format(int(i), class_name, function_name) for i, function_name in enumerate(function_names)])
        str_trees = '\n'.join(functions)
        str_method = AdaBoostClassifier._create_method(model, method_name, str_tree_calls, n_estimators)
        str_class = AdaBoostClassifier._create_class(model, class_name, str_trees, str_method)
        return str_class


    @staticmethod
    def _create_trees(model, method_name):
        method_name = str(method_name)
        n_features = model.n_features_
        n_classes = model.n_classes_

        def _recurse(left, right, threshold, value, features, node, depth):
            out = ''
            indent = '\n' + '    ' * depth
            if threshold[node] != -2.:
                out += indent + 'if (atts[{0}] <= {1:.6f}f) {{'.format(features[node], threshold[node])
                if left[node] != -1.:
                    out += _recurse(left, right, threshold, value, features, left[node], depth + 1)
                out += indent + '} else {'
                if right[node] != -1.:
                    out += _recurse(left, right, threshold, value, features, right[node], depth + 1)
                out += indent + '}'
            else:
                out += ';'.join(
                    [indent + 'classes[{0}] = {1:.6f}f'.format(i, v) for i, v in enumerate(value[node][0])]) + ';'
            return out

        features = [[str(idx) for idx in range(n_features)][i] for i in model.tree_.feature]

        conditions = _recurse(
            model.tree_.children_left,
            model.tree_.children_right,
            model.tree_.threshold,
            model.tree_.value, features, 0, 1)

        out = (
            'public static float[] {0}(float[] atts) {{ \n'
            '    int n_classes = {1}; \n'
            '    float[] classes = new float[n_classes]; \n'
            '    {2} \n\n'
            '    return classes; \n'
            '}}'
        ).format(method_name, n_classes, conditions)  # -> {0}, {1}, {2}
        return str(out)


    @staticmethod
    def _create_method(model, method_name, str_trees, n_estimators):
        print("Ada._create_method()")
        n_classes = model.n_classes_
        out = (
            'public static int {0}(float[] atts) {{ \n'
            '    int n_estimators = {1}; \n'
            '    float[][] preds = new float[n_estimators][]; \n'
            '    {2} \n'
            '    int n_classes = {3}; \n'
            '    float[] classes = new float[n_classes]; \n'
            '    for (int i = 1; i < n_estimators; i++) {{ \n'
            '        for (int j = 0; j < n_classes; j++) {{ \n'
            '            classes[j] += preds[i][j]; \n'
            '        }} \n'
            '    }} \n\n'
            '    int idx = 0; \n'
            '    float val = classes[0]; \n'
            '    for (int i = 1; i < n_classes; i++) {{ \n'
            '        if (classes[i] > val) {{ \n'
            '            idx = i; \n'
            '        }} \n'
            '    }} \n'
            '    return idx; \n'
            '}}').format(method_name, n_estimators, str_trees, n_classes)
        return str(out)


    @staticmethod
    def _create_class(model, class_name, str_trees, str_method):
        n_features = model.estimators_[0].n_features_
        out = (
            'class {0} {{ \n'
            '    {1} \n'
            '    {2} \n'
            '    public static void main(String[] args) {{ \n'
            '        if (args.length == {3}) {{ \n'
            '            float[] atts = new float[args.length]; \n'
            '            for (int i = 0; i < args.length; i++) {{ \n'
            '                atts[i] = Float.parseFloat(args[i]); \n'
            '            }} \n'
            '            System.out.println({0}.predict(atts)); \n'
            '        }} \n'
            '    }} \n'
            '}}').format(class_name, str_trees, str_method, n_features)
        return str(out)
