from Classifier import Classifier


class LinearSVC(Classifier):
    """
    See also
    --------
    sklearn.svm.LinearSVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.LinearSVC.html
    """

    SUPPORT = {'predict': ['c', 'java', 'js']}

    # @formatter:off
    TEMPLATE = {
        'c': {
            'type': ('{0}'),
            'arr': ('{{{0}}}'),
            'arr[]': ('double {name}[{n}] = {{{values}}};'),
            'arr[][]': ('double {name}[{n}][{m}] = {{{values}}};'),
            'method': (
                'int {name} (float atts[{n_features}]) {{ \n'
                '    {coefficients} \n'
                '    {intercepts} \n'
                '    double class_val = -INFINITY; \n'
                '    int class_idx = -1; \n'
                '    for (int i = 0; i < {n_classes}; i++) {{ \n'
                '        double prob = 0.; \n'
                '        for (int j = 0; j < {n_features}; j++) {{ \n'
                '            prob += coefs[i][j] * atts[j]; \n'
                '        }} \n'
                '        if (prob + inters[i] > class_val) {{ \n'
                '            class_val = prob + inters[i]; \n'
                '            class_idx = i; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}}'
            ),
            'class': (
                '#include <stdio.h> \n'
                '#include <math.h> \n\n'
                '{method} \n\n'
                'int main(int argc, const char * argv[]) {{ \n'
                '    float test[{n_features}] = {{1.f, 2.f, 3.f, 4.f}}; \n'
                '    printf("Result: %d\\n", {method_name}(test)); \n'
                '    return 0; \n'
                '}}'
            )
        },
        'java': {
            'type': ('{0}'),
            'arr': ('{{{0}}}'),
            'arr[]': ('double[] {name} = {{{values}}};'),
            'arr[][]': ('double[][] {name} = {{{values}}};'),
            'method': (
                '{i}public static int {name}(float[] atts) {{ \n'
                '{i}    if (atts.length != {n_features}) {{ return -1; }} \n'
                '{i}    {coefficients} \n'
                '{i}    {intercepts} \n'
                '{i}    int class_idx = -1; \n'
                '{i}    double class_val = Double.NEGATIVE_INFINITY; \n'
                '{i}    for (int i = 0; i < {n_classes}; i++) {{ \n'
                '{i}        double prob = 0.; \n'
                '{i}        for (int j = 0; j < {n_features}; j++) {{ \n'
                '{i}            prob += coefs[i][j] * atts[j]; \n'
                '{i}        }} \n'
                '{i}        if (prob + inters[i] > class_val) {{ \n'
                '{i}            class_val = prob + inters[i]; \n'
                '{i}            class_idx = i; \n'
                '{i}        }} \n'
                '{i}    }} \n'
                '{i}    return class_idx; \n'
                '{i}}}'
            ),
            'class': (
                'class {class_name} {{ \n'
                '{method} \n'
                '{i}public static void main(String[] args) {{ \n'
                '{i}    if (args.length == {n_features}) {{ \n'
                '{i}        float[] atts = new float[args.length]; \n'
                '{i}        for (int i = 0, l = args.length; i < l; i++) {{ \n'
                '{i}            atts[i] = Float.parseFloat(args[i]); \n'
                '{i}        }} \n'
                '{i}        System.out.println({class_name}.{method_name}(atts)); \n'
                '{i}    }} \n'
                '{i}}} \n'
                '}}'
            )
        },
        'js': {
            'type': ('{0}'),
            'arr': ('[{0}]'),
            'arr[]': ('var {name} = [{values}];'),
            'arr[][]': ('var {name} = [{values}];'),
            'method': (
                'var {name} = function(atts) {{ \n'
                '    if (atts.length != {n_features}) {{ return -1; }}; \n'
                '    {coefficients} \n'
                '    {intercepts} \n'
                '    var class_idx = -1, \n'
                '        class_val = Number.NEGATIVE_INFINITY, \n'
                '        prob = 0.; \n'
                '    for (var i = 0; i < {n_classes}; i++) {{ \n'
                '        prob = 0.; \n'
                '        for (var j = 0; j < {n_features}; j++) {{ \n'
                '            prob += coefs[i][j] * atts[j]; \n'
                '        }} \n'
                '        if (prob + inters[i] > class_val) {{ \n'
                '            class_val = prob + inters[i]; \n'
                '            class_idx = i; \n'
                '        }} \n'
                '    }} \n'
                '    return class_idx; \n'
                '}};'
            ),
            'class': ('{method}')
        }
    }
    # @formatter:on


    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(LinearSVC, self).__init__(language, method_name, class_name)


    def port(self, model):
        """Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : LinearSVC
            An instance of a trained LinearSVC classifier.
        """
        super(self.__class__, self).port(model)

        self.n_features = len(self.model.coef_[0])
        self.n_classes = len(self.model.classes_)

        if self.method_name == 'predict':
            return self.predict()


    def predict(self):
        """Port the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method.
        """
        return self.create_class(self.create_method())


    def create_method(self):
        """Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """

        # Coefficients:
        coefs = []
        for idx, coef in enumerate(self.model.coef_):
            tmp = [self.temp('type').format(repr(c)) for c in coef]
            tmp = self.temp('arr').format(', '.join(tmp))
            coefs.append(tmp)
        coefs = ', '.join(coefs)
        coefs = self.temp('arr[][]').format(
            name='coefs',
            values=coefs,
            n=self.n_classes,
            m=self.n_features)

        # Intercepts:
        inters = [self.temp('type').format(repr(i)) for i in self.model.intercept_]
        inters = ', '.join(inters)
        inters = self.temp('arr[]').format(
            name='inters',
            values=inters,
            n=self.n_classes)

        return self.temp('method').format(
            name=self.method_name,
            n_features=self.n_features,
            n_classes=self.n_classes,
            coefficients=coefs,
            intercepts=inters,
            i='    ')


    def create_class(self, method):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            class_name=self.class_name,
            method_name=self.method_name,
            method=method,
            n_features=self.n_features,
            i='    ')
