from .. import Classifier


class KNeighborsClassifier(Classifier):
    """
    See also
    --------
    sklearn.neighbors.KNeighborsClassifier

    http://scikit-learn.org/0.18/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    SUPPORTED_METHODS = ['predict']

    # @formatter:off
    TEMPLATES = {
        'java': {
            'type':     '{0}',
            'arr':      '{{{0}}}',
            'arr[]':    'double[] {name} = {{{values}}};',
            'arr[][]':  'double[][] {name} = {{{values}}};',
            'indent':   '    ',
        },
    }
    # @formatter:on

    def __init__(
            self, language='java', method_name='predict', class_name='Tmp'):
        super(KNeighborsClassifier, self).__init__(language, method_name, class_name)

    def port(self, model):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : KNeighborsClassifier
            An instance of a trained KNeighborsClassifier classifier.
        """
        super(self.__class__, self).port(model)

        self.n_classes = len(self.model.classes_)
        self.algorithm = self.model.algorithm
        if self.algorithm is not 'brute':
            from sklearn.neighbors.kd_tree import KDTree
            from sklearn.neighbors.ball_tree import BallTree
            tree = self.model._tree
            if isinstance(tree, (KDTree, BallTree)):
                self.tree = tree

        self.metric = self.model.metric

        print('algorithm', self.model.algorithm)
        print('classes_', self.model.classes_)
        print('metric', self.model.metric)
        print('metric_params', self.model.metric_params)
        print('n_neighbors', self.model.n_neighbors)
        print('radius', self.model.radius)
        print('algorithm', self.model.algorithm)
        print('weights', self.model.weights)

        print('_fit_X', self.model._fit_X)
        print('_y', self.model._y)
        print('_tree', self.model._tree)

        # print('_tree.data', self.model._tree.data)

        # for i in dir(self.model._tree):
        #     print(i, "  ", type(getattr(self.model._tree, i)))

        # self.n_features = len(self.model.coef_[0])
        # self.n_classes = len(self.model.classes_)

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
        # return self.create_class(self.create_method())
        pass

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
