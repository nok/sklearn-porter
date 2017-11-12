# -*- coding: utf-8 -*-

from sklearn_porter.estimator.classifier.Classifier import Classifier


class KNeighborsClassifier(Classifier):
    """
    See also
    --------
    sklearn.neighbors.KNeighborsClassifier

    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
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
        'js': {
            'type':     '{0}',
            'arr':      '[{0}]',
            'arr[]':    'var {name} = [{values}];',
            'arr[][]':  'var {name} = [{values}];',
            'indent':   '    ',
        },
    }
    # @formatter:on

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param estimator : KNeighborsClassifier
            An instance of a trained AdaBoostClassifier estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(KNeighborsClassifier, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)

        if estimator.weights != 'uniform':
            msg = "Only 'uniform' weights are supported for this classifier."
            raise NotImplementedError(msg)

        self.estimator = estimator

    def export(self, class_name, method_name, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name: string, default: 'Brain'
            The name of the class in the returned result.
        :param method_name: string, default: 'predict'
            The name of the method in the returned result.

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

        self.metric = est.metric
        self.n_classes = len(est.classes_)
        self.n_templates = len(est._fit_X)  # pylint: disable=W0212
        self.n_features = len(est._fit_X[0])  # pylint: disable=W0212
        self.n_neighbors = est.n_neighbors

        self.algorithm = est.algorithm
        self.power_param = est.p

        if self.algorithm != 'brute':
            from sklearn.neighbors.kd_tree import KDTree  # pylint: disable-msg=E0611
            from sklearn.neighbors.ball_tree import BallTree  # pylint: disable-msg=E0611
            tree = est._tree  # pylint: disable=W0212
            if isinstance(tree, (KDTree, BallTree)):
                self.tree = tree

        if self.target_method == 'predict':
            return self.predict()

    def predict(self):
        """
        Transpile the predict method.

        Returns
        -------
        :return : string
            The transpiled predict method as string.
        """
        return self.create_class(self.create_method())

    def create_method(self):
        """
        Build the estimator method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """

        # Distance computation
        metric_name = '.'.join(['metric', self.metric])
        distance_comp = self.temp(metric_name, n_indents=1, skipping=True)

        temp_method = self.temp('method.predict', n_indents=1, skipping=True)
        out = temp_method.format(class_name=self.class_name,
                                 method_name=self.method_name,
                                 distance_computation=distance_comp)
        return out

    def create_class(self, method):
        """
        Build the estimator class.

        Returns
        -------
        :return out : string
            The built class as string.
        """

        temp_type = self.temp('type')
        temp_arr = self.temp('arr')
        temp_arr_ = self.temp('arr[]')
        temp_arr__ = self.temp('arr[][]')

        # Templates
        temps = []
        for atts in enumerate(self.estimator._fit_X):  # pylint: disable=W0212
            tmp = [temp_type.format(self.repr(a)) for a in atts[1]]
            tmp = temp_arr.format(', '.join(tmp))
            temps.append(tmp)
        temps = ', '.join(temps)
        temps = temp_arr__.format(type='double', name='X', values=temps,
                                  n=self.n_templates, m=self.n_features)

        # Classes
        classes = self.estimator._y  # pylint: disable=W0212
        classes = [temp_type.format(int(c)) for c in classes]
        classes = ', '.join(classes)
        classes = temp_arr_.format(type='int', name='y', values=classes,
                                   n=self.n_templates)

        temp_class = self.temp('class')
        out = temp_class.format(class_name=self.class_name,
                                method_name=self.method_name, method=method,
                                n_features=self.n_features, X=temps, y=classes,
                                n_neighbors=self.n_neighbors,
                                n_templates=self.n_templates,
                                n_classes=self.n_classes,
                                power=self.power_param)
        return out
