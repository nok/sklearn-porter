from .. import Classifier


class LinearSVC(Classifier):
    """
    See also
    --------
    sklearn.svm.LinearSVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.LinearSVC.html
    """

    SUPPORT = {'predict': ['c', 'go', 'java', 'js']}

    # @formatter:off
    TEMPLATE = {
        'c': {
            'type':     ('{0}'),
            'arr':      ('{{{0}}}'),
            'arr[]':    ('double {name}[{n}] = {{{values}}};'),
            'arr[][]':  ('double {name}[{n}][{m}] = {{{values}}};'),
            'indent':   ('    '),
        },
        'go': {
            'type':     ('{0}'),
            'arr':      ('{{{0}}}'),
            'arr[]':    ('{name} := []float64{{{values}}}'),
            'arr[][]':  ('{name} := [][]float64{{{values}}}'),
            'indent':   ('\t'),
        },
        'java': {
            'type':     ('{0}'),
            'arr':      ('{{{0}}}'),
            'arr[]':    ('double[] {name} = {{{values}}};'),
            'arr[][]':  ('double[][] {name} = {{{values}}};'),
            'indent':   ('    '),
        },
        'js': {
            'type':     ('{0}'),
            'arr':      ('[{0}]'),
            'arr[]':    ('var {name} = [{values}];'),
            'arr[][]':  ('var {name} = [{values}];'),
            'indent':   ('    '),
        }
    }
    # @formatter:on


    def __init__(
            self, language='java', method_name='predict', class_name='Tmp'):
        super(LinearSVC, self).__init__(language, method_name, class_name)


    def port(self, model):
        """
        Port a trained model to the syntax of a chosen programming language.

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
        """
        Port the predict method.

        Returns
        -------
        :return: out : string
            The ported predict method.
        """
        return self.create_class(self.create_method())


    def create_method(self):
        """
        Build the model method or function.

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
            name='coefs', values=coefs, n=self.n_classes, m=self.n_features)

        # Intercepts:
        inters = self.model.intercept_
        inters = [self.temp('type').format(repr(i)) for i in inters]
        inters = ', '.join(inters)
        inters = self.temp('arr[]').format(
            name='inters', values=inters, n=self.n_classes)

        return self.temp('method', indentation=1, skipping=True).format(
            name=self.method_name, n_features=self.n_features,
            n_classes=self.n_classes, coefficients=coefs,
            intercepts=inters)


    def create_class(self, method):
        """
        Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            class_name=self.class_name, method_name=self.method_name,
            method=method, n_features=self.n_features)
