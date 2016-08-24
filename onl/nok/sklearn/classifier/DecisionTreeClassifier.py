from Classifier import Classifier


class DecisionTreeClassifier(Classifier):
    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(self.__class__, self).__init__(language, method_name, class_name)
        if method_name not in ['predict']:
            msg = 'The classifier does not support the given method.'
            raise ValueError(msg)

    def port(self, model):
        if self.method_name == 'predict':
            return self.predict(model)

    def predict(self, model):
        str_method = DecisionTreeClassifier._create_method(model, 'predict')
        str_class = DecisionTreeClassifier._create_class(model, self.class_name)
        return str_class.format(str_method)

    @staticmethod
    def _create_method(model, method_name):
        method_name = str(method_name)
        n_features = model.n_features_
        n_classes = model.n_classes_

        # @formatter:off
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
        # @formatter:on

        features = [[str(idx) for idx in range(n_features)][i] for i in model.tree_.feature]
        conditions = _recurse(
            model.tree_.children_left,
            model.tree_.children_right,
            model.tree_.threshold,
            model.tree_.value, features, 0, 1)

        # @formatter:off
        out = (
            'public static int {0}(float[] atts) {{ \n'
            '    int n_classes = {1}; \n'
            '    int[] classes = new int[n_classes]; \n'
            '    {2} \n\n'
            '    int idx = 0; \n'
            '    int val = classes[0]; \n'
            '    for (int i = 1; i < n_classes; i++) {{ \n'
            '        if (classes[i] > val) {{ \n'
            '            idx = i; \n'
            '            val = classes[i]; \n'
            '        }} \n'
            '    }} \n'
            '    return idx; \n'
            '}}'
        ).format(method_name, n_classes, conditions)  # -> {0}, {1}, {2}
        # @formatter:on
        return str(out)

    @staticmethod
    def _create_class(model, class_name):
        n_features = model.n_features_
        out = (
            'class {0} {{{{ \n'
            '    {{0}} \n'
            '    public static void main(String[] args) {{{{ \n'
            '        if (args.length == {1}) {{{{ \n'
            '            float[] atts = new float[args.length]; \n'
            '            for (int i = 0; i < args.length; i++) {{{{ \n'
            '                atts[i] = Float.parseFloat(args[i]); \n'
            '            }}}} \n'
            '            System.out.println({0}.predict(atts)); \n'
            '        }}}} \n'
            '    }}}} \n'
            '}}}}').format(class_name, n_features)  # -> {0}, {1}
        return str(out)
