# from onl.nok.sklearn.export.classifier.Classifier import Classifier


class DecisionTreeClassifier():

    @staticmethod
    def _get_supported_methods():
        return [
            'predict'
        ]

    @staticmethod
    def export(model, method):
        # TODO: Add method check
        # if not DecisionTreeClassifier._is_supported_method(method):
        #     return
        return DecisionTreeClassifier.predict(model)

    @staticmethod
    def _is_supported_method(method):
        support = method in DecisionTreeClassifier._get_supported_methods()
        if not support:
            raise ValueError('The classifier does not support the method.')
        return support

    @staticmethod
    def predict(model):
        n_features = model.n_features_
        n_classes = model.n_classes_
        method_name = 'predict'

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
                    [indent + 'classes[{0}] = {1}'.format(i, int(v)) for i, v in enumerate(value[node][0])]) + ';'
            return out

        features = [[str(idx) for idx in range(n_features)][i] for i in model.tree_.feature]
        conditions = _recurse(
            model.tree_.children_left,
            model.tree_.children_right,
            model.tree_.threshold,
            model.tree_.value, features, 0, 1)

        source = (
            'public static int {0} (float[] atts) {{ \n'
            '    int n_classes = {1}; \n'
            '    int[] classes = new int[n_classes]; \n'
            '    {2} \n\n'
            '    int idx = 0; \n'
            '    int val = classes[0]; \n'
            '    for (int i = 1; i < n_classes; i++) {{ \n'
            '        if (classes[i] > val) {{ \n'
            '            idx = i; \n'
            '        }} \n'
            '    }} \n'
            '    return idx; \n'
            '}}'
        ).format(str(method_name), n_classes, conditions)

        return source
