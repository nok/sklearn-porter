# -*- coding: utf-8 -*-

import types
from ..Classifier import Classifier


class SVC(Classifier):
    """
    See also
    --------
    sklearn.svm.SVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.SVC.html
    """

    SUPPORTED_METHODS = ['predict']

    # @formatter:off
    TEMPLATES = {
        'c': {
            'type':     '{0}',
            'arr':      '{{{0}}}',
            'arr[]':    '{type} {name}[{n}] = {{{values}}};',
            'arr[][]':  '{type} {name}[{n}][{m}] = {{{values}}};',
            'indent':   '    ',
        },
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
        'php': {
            'type':     '{0}',
            'arr':      '[{0}]',
            'arr[]':    '${name} = [{values}];',
            'arr[][]':  '${name} = [{values}];',
            'indent':   '    ',
        },
        'ruby': {
            'type':     '{0}',
            'arr':      '[{0}]',
            'arr[]':    '{name} = [{values}]',
            'arr[][]':  '{name} = [{values}]',
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
        :param estimator : AdaBoostClassifier
            An instance of a trained SVC estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(SVC, self).__init__(estimator, target_language=target_language,
                                  target_method=target_method, **kwargs)
        self.estimator = estimator

    def export(self, class_name, method_name):
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

        # Templates of primitive data types:
        temp_type = self.temp('type')
        temp_arr = self.temp('arr')
        temp_arr_ = self.temp('arr[]')
        temp_arr__ = self.temp('arr[][]')

        # Estimator:
        est = self.estimator
        params = est.get_params()

        # Check kernel type:
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if params['kernel'] not in kernels:
            msg = 'The kernel type is not supported.'
            raise ValueError(msg)

        # Check rbf gamma value:
        if params['kernel'] == 'rbf' and params['gamma'] == 'auto':
            msg = ('The classifier gamma value have to '
                   'be set (currently it is \'auto\').')
            raise ValueError(msg)
        self.params = params

        self.n_features = len(est.support_vectors_[0])
        self.svs_rows = est.n_support_
        self.n_svs_rows = len(est.n_support_)

        self.weights = self.temp('arr[]', skipping=True).format(
            type='int', name='weights', values=', '.join([str(e) for e in
                                                          self.svs_rows]),
            n=len(self.svs_rows))
        self.n_weights = len(self.svs_rows)

        self.n_classes = len(est.classes_)
        self.is_binary = self.n_classes == 2
        self.prefix = 'binary' if self.is_binary else 'multi'

        # Support vectors:
        vectors = []
        for vector in est.support_vectors_:
            _vectors = [temp_type.format(self.repr(v)) for v in vector]
            _vectors = temp_arr.format(', '.join(_vectors))
            vectors.append(_vectors)
        vectors = ', '.join(vectors)
        vectors = self.temp('arr[][]', skipping=True).format(
            type='double', name='vectors', values=vectors,
            n=len(est.support_vectors_), m=len(est.support_vectors_[0]))
        self.vectors = vectors
        self.n_vectors = len(est.support_vectors_)

        # Coefficients:
        coeffs = []
        for coeff in est.dual_coef_:
            _coeffs = [temp_type.format(self.repr(c)) for c in coeff]
            _coeffs = temp_arr.format(', '.join(_coeffs))
            coeffs.append(_coeffs)
        coeffs = ', '.join(coeffs)
        coeffs = temp_arr__.format(type='double', name='coefficients',
                                   values=coeffs, n=len(est.dual_coef_),
                                   m=len(est.dual_coef_[0]))
        self.coefficients = coeffs
        self.n_coefficients = len(est.dual_coef_)

        # Interceptions:
        inters = [temp_type.format(self.repr(i)) for i in est._intercept_]
        inters = ', '.join(inters)
        inters = temp_arr_.format(type='double', name='intercepts',
                                  values=inters, n=len(est._intercept_))
        self.intercepts = inters
        self.n_intercepts = len(est._intercept_)

        # Kernel:
        self.kernel = str(params['kernel'])[0] if self.target_language == 'c'\
            else str(params['kernel'])
        self.gamma = self.repr(self.params['gamma'])
        self.coef0 = self.repr(self.params['coef0'])
        self.degree = self.repr(self.params['degree'])

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
        self.method = self.create_method()
        output = self.create_class()
        return output

    def create_method(self):
        """
        Build the estimator method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        n_indents = 1 if self.target_language in ['java', 'js',
                                                  'php', 'ruby'] else 0
        method = self.temp('method', n_indents=n_indents,
                           skipping=True).format(**self.__dict__)
        return method

    def create_class(self):
        """
        Build the estimator class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        temp_class = self.temp('class')
        out = temp_class.format(**self.__dict__)
        return out
