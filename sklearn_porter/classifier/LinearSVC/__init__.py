# -*- coding: utf-8 -*-

from ..Classifier import Classifier


class LinearSVC(Classifier):
    """
    See also
    --------
    sklearn.svm.LinearSVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.LinearSVC.html
    """

    SUPPORTED_METHODS = ['predict']

    # @formatter:off
    TEMPLATES = {
        'c': {
            'init':     '{type} {name} = {value};',
            'type':     '{0}',
            'arr':      '{{{0}}}',
            'arr[]':    'double {name}[{n}] = {{{values}}};',
            'arr[][]':  'double {name}[{n}][{m}] = {{{values}}};',
            'indent':   '    ',
        },
        'go': {
            'init':     '{name} := {value}',
            'type':     '{0}',
            'arr':      '{{{0}}}',
            'arr[]':    '{name} := []float64{{{values}}}',
            'arr[][]':  '{name} := [][]float64{{{values}}}',
            'indent':   '\t',
        },
        'java': {
            'init':     '{type} {name} = {value};',
            'type':     '{0}',
            'arr':      '{{{0}}}',
            'arr[]':    'double[] {name} = {{{values}}};',
            'arr[][]':  'double[][] {name} = {{{values}}};',
            'indent':   '    ',
        },
        'js': {
            'init':     'var {name} = {value};',
            'type':     '{0}',
            'arr':      '[{0}]',
            'arr[]':    'var {name} = [{values}];',
            'arr[][]':  'var {name} = [{values}];',
            'indent':   '    ',
        },
        'php': {
            'init':     '${name} = {value};',
            'type':     '{0}',
            'arr':      '[{0}]',
            'arr[]':    '${name} = [{values}];',
            'arr[][]':  '${name} = [{values}];',
            'indent':   '    ',
        },
        'ruby': {
            'init':     '{name} = {value}',
            'type':     '{0}',
            'arr':      '[{0}]',
            'arr[]':    '{name} = [{values}]',
            'arr[][]':  '{name} = [{values}]',
            'indent':   '    ',
        }
    }
    # @formatter:on

    def __init__(self, model, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : LinearSVC
            An instance of a trained AdaBoostClassifier model.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(LinearSVC, self).__init__(model, target_language=target_language,
                                        target_method=target_method, **kwargs)
        self.model = model
        self.n_features = len(model.coef_[0])
        self.n_classes = len(model.classes_)
        self.is_binary = self.n_classes == 2
        self.prefix = 'binary.' if self.is_binary else 'multi.'

    def export(self, class_name="Brain", method_name="predict", use_repr=True):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name: string, default: 'Brain'
            The name of the class in the returned result.
        :param method_name: string, default: 'predict'
            The name of the method in the returned result.
        :param use_repr : bool, default True
            Whether to use repr() for floating-point values or not.

        Returns
        -------
        :return : string
            The transpiled algorithm with the defined placeholders.
        """
        self.class_name = class_name
        self.method_name = method_name
        self.use_repr = use_repr
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
        Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """

        # Coefficients:
        if self.is_binary:
            coefs = self.model.coef_[0]
            coefs = [self.temp('type').format(self.repr(c)) for c in coefs]
            coefs = ', '.join(coefs)
            coefs = self.temp('arr[]').format(
                name='coefs', values=coefs, n=self.n_features)
        else:
            coefs = []
            for coef in self.model.coef_:
                tmp = [self.temp('type').format(self.repr(c)) for c in coef]
                tmp = self.temp('arr').format(', '.join(tmp))
                coefs.append(tmp)
            coefs = ', '.join(coefs)
            coefs = self.temp('arr[][]').format(
                name='coefs', values=coefs, n=self.n_classes, m=self.n_features)

        # Intercepts:
        if self.is_binary:
            inters = self.model.intercept_[0]
            inters = self.temp('init').format(type='double', name='inters',
                                              value=self.repr(inters))
        else:
            inters = self.model.intercept_
            inters = [self.temp('type').format(self.repr(i)) for i in inters]
            inters = ', '.join(inters)
            inters = self.temp('arr[]').format(
                name='inters', values=inters, n=self.n_classes)

        # Indentation
        n_indents = 0 if self.target_language in ['c', 'go'] else 1

        return self.temp(self.prefix + 'method',
                         n_indents=n_indents, skipping=True).format(
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
