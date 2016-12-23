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
            'arr[]':    '{type}[] {name} = {{{values}}};',
            'arr[][]':  '{type}[][] {name} = {{{values}}};',
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
        self.n_templates = len(self.model._fit_X)
        self.n_features = len(self.model._fit_X[0])

        self.algorithm = self.model.algorithm
        self.power_param = self.model.p

        if self.algorithm is not 'brute':
            from sklearn.neighbors.kd_tree import KDTree
            from sklearn.neighbors.ball_tree import BallTree
            tree = self.model._tree
            if isinstance(tree, (KDTree, BallTree)):
                self.tree = tree

        self.metric = self.model.metric

        # print('algorithm', self.model.algorithm)
        # print('classes_', self.model.classes_)
        # print('metric', self.model.metric)
        # print('metric_params', self.model.metric_params)
        # print('n_neighbors', self.model.n_neighbors)
        # print('radius', self.model.radius)
        # print('algorithm', self.model.algorithm)
        # print('weights', self.model.weights)

        # print('_fit_X', self.model._fit_X)
        # print('_y', self.model._y)
        # print('_tree', self.model._tree)
        # print('p', self.model.p)

        # print('_tree.data', self.model._tree.data)

        # for i in dir(self.model):
        #     print(i, "  ", type(getattr(self.model, i)))

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
        return self.create_class(self.create_method())

    def create_method(self):
        """
        Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """

        # Distance computation
        metric_name = '.'.join(['metric', self.metric])
        distance_comp = self.temp(
            metric_name, indentation=1, skipping=True)

        # Templates
        temps = []
        for atts in enumerate(self.model._fit_X):
            tmp = [self.temp('type').format(repr(a)) for a in atts[1]]
            tmp = self.temp('arr').format(', '.join(tmp))
            temps.append(tmp)
        temps = ', '.join(temps)
        temps = self.temp('arr[][]').format(
            type='double',
            name='X',
            values=temps,
            n=self.n_templates,
            m=self.n_features)

        # Classes
        classes = self.model._y
        classes = [self.temp('type').format(int(c)) for c in classes]
        classes = ', '.join(classes)
        classes = self.temp('arr[]').format(
            type='int',
            name='y',
            values=classes,
            n=self.n_templates)

        return self.temp('method.predict', indentation=1, skipping=True).format(
            method_name=self.method_name,
            class_name=self.class_name,
            n_templates=self.n_templates,
            n_features=self.n_features,
            n_classes=self.n_classes,
            distance_computation=distance_comp,
            power=self.power_param,
            X=temps,
            y=classes)

    def create_class(self, method):
        """
        Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            class_name=self.class_name,
            method_name=self.method_name,
            method=method,
            n_features=self.n_features)
