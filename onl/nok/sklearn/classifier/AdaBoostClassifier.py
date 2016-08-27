import sklearn
import copy

from Classifier import Classifier


class AdaBoostClassifier(Classifier):
    """
    See also
    --------
    sklearn.ensemble.AdaBoostClassifier

    http://scikit-learn.org/0.17/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    """

    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(self.__class__, self).__init__(language, method_name, class_name)
        if method_name not in ['predict']:
            msg = 'The classifier does not support the given method.'
            raise ValueError(msg)


    def port(self, model):
        """Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : AdaBoostClassifier
            An instance of a trained AdaBoostClassifier model.
        """

        # Check the used algorithm type:
        if model.algorithm not in ('SAMME.R'):
            msg = 'The classifier does not support the given algorithm %s.'
            raise ValueError(msg, model.algorithm)

        # Check type of base estimators:
        if not isinstance(model.base_estimator, sklearn.tree.tree.DecisionTreeClassifier):
            msg = 'The classifier does not support the given base estimator %s.'
            raise ValueError(msg, model.base_estimator)

        # Check number of base estimators:
        if not model.n_estimators > 0:
            msg = 'The classifier has not any base estimators.'
            raise ValueError(msg)

        self.model = model
        self.n_classes = model.n_classes_

        self.models = []
        self.weights = []
        self.n_estimators = 0
        for idx in range(self.model.n_estimators):
            weight = float(self.model.estimator_weights_[idx])
            if weight > 0:
                self.models.append(copy.deepcopy(self.model.estimators_[idx]))
                self.weights.append(copy.deepcopy(self.model.estimator_weights_[idx]))
                self.n_estimators += 1
                self.n_features = self.model.estimators_[idx].n_features_

        if self.method_name == 'predict':
            return self.predict()


    def predict(self):
        """Port the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method.
        """
        str_class = self.create_class()
        str_methods = self.create_methods()
        out = str_class.format(str_methods)
        return out


    def parse_tree(self, L, R, T, value, features, node, depth):
        """Port the structure of the model.

        Parameters
        ----------
        :param L : object
            The left children node.
        :param R : object
            The left children node.
        :param T : object
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
            The ported structure of the tree model.
        """
        out = ''
        indent = '\n' + '    ' * depth
        # @formatter:off
        if T[node] != -2.:
            template = {
                # @formatter:off
                'java': ('if (atts[{0}] <= {1:.16}f) {{'),
                'js': ('if (atts[{0}] <= {1:.16}) {{')
                # @formatter:on
            }
            out += indent + template[self.language].format(features[node], T[node])
            if L[node] != -1.:
                out += self.parse_tree(L, R, T, value, features, L[node], depth + 1)
            out += indent + '} else {'
            if R[node] != -1.:
                out += self.parse_tree(L, R, T, value, features, R[node], depth + 1)
            out += indent + '}'
        else:
            classes = []
            template = {
                # @formatter:off
                'java': (indent + 'classes[{0}] = {1:.16}f'),
                'js': (indent + 'classes[{0}] = {1:.16}')
                # @formatter:on
            }
            for idx, val in enumerate(value[node][0]):
                classes.append(template[self.language].format(idx, val))
            out += ';'.join(classes) + ';'
        return out


    def create_tree(self, model_index, model):
        """Port a method for a single tree.

        Parameters
        ----------
        :param model_index : int
            The model index.
        :param model : DecisionTreeClassifier
            The model.

        Returns
        -------
        :return : string
            The created method.
        """
        feature_indices = []
        for idx in model.tree_.feature:
            feature_indices.append([str(jdx) for jdx in range(model.n_features_)][idx])

        tree_branches = self.parse_tree(
            model.tree_.children_left,
            model.tree_.children_right,
            model.tree_.threshold,
            model.tree_.value,
            feature_indices, 0, 1)

        template = {
            'java': (
                # @formatter:off
                'public static float[] {1}_{0}(float[] atts) {{ \n'
                '    float[] classes = new float[{2}];'
                '    {3} \n\n'
                '    return classes; \n'
                '}}'
                # @formatter:on
            ),
            'js': (
                # @formatter:off
                'var {1}_{0} = function(atts) {{ \n'
                '    var classes = new Array({2});'
                '    {3} \n\n'
                '    return classes; \n'
                '}};'
                # @formatter:on
            )
        }

        out = template[self.language].format(
            str(model_index),
            self.method_name,
            self.n_classes,
            tree_branches)
        return out


    def create_methods(self):
        """Build the model methods or functions.

        Returns
        -------
        :return out : string
            The built methods as merged string.
        """
        # Generate method or function names:
        fn_names = []
        template = {
            'java': (
                # @formatter:off
                # preds[0] = Tmp.predict_0(atts);
                'preds[{0}] = {1}.{2}(atts);'
                # @formatter:on
            ),
            'js': (
                # @formatter:off
                # preds[0] = predict_0(atts);
                'preds[{0}] = {2}(atts);'
                # @formatter:on
            )
        }
        suffix = ("_{0:0" + str(len(str(self.n_estimators))) + "d}")
        for idx, model in enumerate(self.models):
            fn_name = self.method_name + suffix.format(idx)
            fn_name = template[self.language].format(idx, self.class_name, fn_name)
            fn_names.append(fn_name)

        # Generate related trees:
        fns = []
        for idx, model in enumerate(self.models):
            tree = self.create_tree(idx, model)
            fns.append(tree)

        template = {
            'java': (
                # @formatter:off
                '{0}'
                'public static int {1}(float[] atts) {{ \n'
                '    int n_estimators = {2}; \n'
                '    int n_classes = {3}; \n\n'
                '    float[][] preds = new float[n_estimators][]; \n'
                '    {4} \n\n'
                '    int i, j; \n'
                '    float normalizer, sum; \n'
                '    for (i = 0; i < n_estimators; i++) {{ \n'
                '        normalizer = 0.f; \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            normalizer += preds[i][j]; \n'
                '        }} \n'
                '        if (normalizer == 0.f) {{ \n'
                '            normalizer = 1.0f; \n'
                '        }} \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            preds[i][j] = preds[i][j] / normalizer; \n'
                '            if (preds[i][j] < 0.000000000000000222044604925f) {{ \n'
                '                preds[i][j] = 0.000000000000000222044604925f; \n'
                '            }} \n'
                '            preds[i][j] = (float) Math.log(preds[i][j]); \n'
                '        }} \n'
                '        sum = 0.0f; \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            sum += preds[i][j]; \n'
                '        }} \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            preds[i][j] = (n_classes - 1) * (preds[i][j] - (1.f / n_classes) * sum); \n'
                '        }} \n'
                '    }} \n'
                '    float[] classes = new float[n_classes]; \n'
                '    for (i = 0; i < n_estimators; i++) {{ \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            classes[j] += preds[i][j]; \n'
                '        }} \n'
                '    }} \n'
                '    int idx = 0; \n'
                '    float val = Float.NEGATIVE_INFINITY; \n'
                '    for (i = 0; i < n_classes; i++) {{ \n'
                '        if (classes[i] > val) {{ \n'
                '            idx = i; \n'
                '            val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return idx; \n'
                '}}'
                # @formatter:on
            ),
            'js': (
                # @formatter:off
                '{0} \n'
                'var {1} = function(atts) {{ \n'
                '    var n_estimators = {2}, \n'
                '        preds = new Array(n_estimators), \n'
                '        n_classes = {3}, \n'
                '        classes = new Array(n_classes), \n'
                '        normalizer, sum, idx, val, \n'
                '        i, j; \n\n'
                '    {4} \n\n'
                '    for (i = 0; i < n_estimators; i++) {{ \n'
                '        normalizer = 0.; \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            normalizer += preds[i][j]; \n'
                '        }} \n'
                '        if (normalizer == 0.) {{ \n'
                '            normalizer = 1.0; \n'
                '        }} \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            preds[i][j] = preds[i][j] / normalizer; \n'
                '            if (preds[i][j] < 0.000000000000000222044604925) {{ \n'
                '                preds[i][j] = 0.000000000000000222044604925; \n'
                '            }} \n'
                '            preds[i][j] = Math.log(preds[i][j]); \n'
                '        }} \n'
                '        sum = 0.0; \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            sum += preds[i][j]; \n'
                '        }} \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            preds[i][j] = (n_classes - 1) * (preds[i][j] - (1. / n_classes) * sum); \n'
                '        }} \n'
                '    }} \n'
                '    for (i = 0; i < n_classes; i++) {{ \n'
                '        classes[i] = 0.0; \n'
                '    }} \n'
                '    for (i = 0; i < n_estimators; i++) {{ \n'
                '        for (j = 0; j < n_classes; j++) {{ \n'
                '            classes[j] += preds[i][j]; \n'
                '        }} \n'
                '    }} \n'
                '    idx = -1; \n'
                '    val = Number.NEGATIVE_INFINITY; \n'
                '    for (i = 0; i < n_classes; i++) {{ \n'
                '        if (classes[i] > val) {{ \n'
                '            idx = i; \n'
                '            val = classes[i]; \n'
                '        }} \n'
                '    }} \n'
                '    return idx; \n'
                '}};'
                # @formatter:on
            )
        }
        # Merge generated content:
        out = template[self.language].format(
            '\n'.join(fns),
            self.method_name,
            self.n_estimators,
            self.n_classes,
            '\n'.join(fn_names))
        return out


    def create_class(self):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        template = {
            'java': (
                # @formatter:off
                'class {0} {{{{ \n'
                '    {{0}} \n'
                '    public static void main(String[] args) {{{{ \n'
                '        if (args.length == {2}) {{{{ \n'
                '            float[] atts = new float[args.length]; \n'
                '            for (int i = 0, l = args.length; i < l; i++) {{{{ \n'
                '                atts[i] = Float.parseFloat(args[i]); \n'
                '            }}}} \n'
                '            System.out.println({0}.{1}(atts)); \n'
                '        }}}} \n'
                '    }}}} \n'
                '}}}}'
                # @formatter:on
            ),
            'js': (
                # @formatter:off
                '{{0}}'
                # @formatter:on
            )
        }
        out = template[self.language].format(
            self.class_name,
            self.method_name,
            self.n_features)
        return out
