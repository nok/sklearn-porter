# -*- coding: utf-8 -*-

from ...Template import Template


class SVC(Template):
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
            'arr[]':    '{type} {name}[] = {{{values}}};',
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
    }
    # @formatter:on

    def __init__(self, model, target_language='java', target_method='predict',
                 **kwargs):
        super(SVC, self).__init__(model, target_language=target_language,
                                  target_method=target_method, **kwargs)
        self.model = model

        params = model.get_params()
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

        self.svs = model.support_vectors_
        self.n_svs = len(model.support_vectors_[0])
        self.svs_rows = model.n_support_
        self.n_svs_rows = len(model.n_support_)
        self.coeffs = model.dual_coef_
        self.inters = model._intercept_
        self.n_inters = len(model._intercept_)
        self.classes = model.classes_
        self.n_classes = len(model.classes_)

    def export(self, class_name, method_name):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : SVC
            An instance of a trained SVC classifier.
        """
        self.class_name = class_name
        self.method_name = method_name
        if self.target_method == 'predict':
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
        out = '\n'

        # Number of support vectors:
        n_svs = [self.temp('type').format(repr(v)) for v in self.svs_rows]
        n_svs = ', '.join(n_svs)
        out += self.temp('arr[]').format(type='int', name='n_svs', values=n_svs)
        out += '\n'

        # Support vectors:
        vectors = []
        for vidx, vector in enumerate(self.svs):
            _vectors = [self.temp('type').format(repr(v)) for v in vector]
            _vectors = self.temp('arr').format(', '.join(_vectors))
            vectors.append(_vectors)
        vectors = ', '.join(vectors)
        out += self.temp('arr[][]', skipping=True).format(
            type='double', name='svs', values=vectors,
            n=len(self.svs), m=self.n_svs)
        out += '\n'

        # Coefficients:
        coeffs = []
        for cidx, coeff in enumerate(self.coeffs):
            _coeffs = [self.temp('type').format(repr(c)) for c in coeff]
            _coeffs = self.temp('arr').format(', '.join(_coeffs))
            coeffs.append(_coeffs)
        coeffs = ', '.join(coeffs)
        out += self.temp('arr[][]').format(
            type='double', name='coeffs', values=coeffs,
            n=len(self.coeffs), m=len(self.coeffs[0]))
        out += '\n'

        # Interceptions:
        inters = [self.temp('type').format(repr(i)) for i in self.inters]
        inters = ', '.join(inters)
        out += self.temp('arr[]').format(
            type='double', name='inters', values=inters)
        out += '\n'

        # Classes:
        classes = [self.temp('type').format(repr(c)) for c in self.classes]
        classes = ', '.join(classes)
        out += self.temp('arr[]').format(
            type='int', name='classes', values=classes)
        out += '\n'

        # Kernels:
        if self.params['kernel'] == 'rbf':
            out += self.temp('kernel.rbf').format(
                len(self.svs), self.n_svs, repr(self.params['gamma']))
        elif self.params['kernel'] == 'poly':
            out += self.temp('kernel.poly').format(
                len(self.svs), self.n_svs, repr(self.params['gamma']),
                repr(self.params['coef0']), repr(self.params['degree']))
        elif self.params['kernel'] == 'sigmoid':
            out += self.temp('kernel.sigmoid').format(
                len(self.svs), self.n_svs, repr(self.params['gamma']),
                repr(self.params['coef0']), repr(self.params['degree']))
        elif self.params['kernel'] == 'linear':
            out += self.temp('kernel.linear').format(
                len(self.svs), self.n_svs)
        out += '\n'

        # Decicion:
        out += self.temp('starts').format(self.n_svs_rows)
        out += self.temp('ends').format(self.n_svs_rows)
        out += self.temp('decicions').format(self.n_inters, self.n_svs_rows)
        out += self.temp('classes').format(self.n_inters, self.n_classes)
        n_indents = 0 if self.target_language in ['java', 'js', 'php'] else 1
        out = self.indent(out, n_indents=2-n_indents)
        return self.temp('method', n_indents=1-n_indents, skipping=True).format(
            method_name=self.method_name, decicion=out)

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
            method=method, n_features=self.n_svs)
