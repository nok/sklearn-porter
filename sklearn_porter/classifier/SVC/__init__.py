from .. import Classifier


class SVC(Classifier):
    """
    See also
    --------
    sklearn.svm.SVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.SVC.html
    """

    SUPPORT = {'predict': ['java', 'js']}

    # @formatter:off
    TEMPLATE = {
        'java': {
            'type':     ('{0}'),
            'arr':      ('{{{0}}}'),
            'arr[]':    ('\n{type}[] {name} = {{{values}}};'),
            'arr[][]':  ('\n{type}[][] {name} = {{{values}}};'),
            'indent':   ('    '),
        },
        'js': {
            'type':     ('{0}'),
            'arr':      ('[{0}]'),
            'arr[]':    ('\nvar {name} = [{values}];'),
            'arr[][]':  ('\nvar {name} = [{values}];'),
            'indent':   ('    '),
        }
    }
    # @formatter:on


    def __init__(
            self, language='java', method_name='predict', class_name='Tmp'):
        super(SVC, self).__init__(language, method_name, class_name)


    def port(self, model):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : SVC
            An instance of a trained SVC classifier.
        """
        super(self.__class__, self).port(model)

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
        self.classes = model.classes_
        self.n_classes = len(model.classes_)

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
        str = ''

        # Support vectors:
        vectors = []
        for vidx, vector in enumerate(self.svs):
            _vectors = [self.temp('type').format(repr(v)) for v in vector]
            _vectors = self.temp('arr').format(', '.join(_vectors))
            vectors.append(_vectors)
        vectors = ', '.join(vectors)
        str += self.temp('arr[][]').format(
            type='double', name='svs', values=vectors)

        # Coefficients:
        coeffs = []
        for cidx, coeff in enumerate(self.coeffs):
            _coeffs = [self.temp('type').format(repr(c)) for c in coeff]
            _coeffs = self.temp('arr').format(', '.join(_coeffs))
            coeffs.append(_coeffs)
        coeffs = ', '.join(coeffs)
        str += self.temp('arr[][]').format(
            type='double', name='coeffs', values=coeffs)

        # Interceptions:
        inters = [self.temp('type').format(repr(i)) for i in self.inters]
        inters = ', '.join(inters)
        str += self.temp('arr[]').format(
            type='double', name='inters', values=inters)

        # Classes:
        classes = [self.temp('type').format(repr(c)) for c in self.classes]
        classes = ', '.join(classes)
        str += self.temp('arr[]').format(
            type='int', name='classes', values=classes) + '\n'

        # Kernels:
        if self.params['kernel'] == 'rbf':
            str += self.temp('kernel.rbf').format(
                len(self.svs), self.n_svs, repr(self.params['gamma']))
        elif self.params['kernel'] == 'poly':
            str += self.temp('kernel.poly').format(
                len(self.svs), self.n_svs, repr(self.params['gamma']),
                repr(self.params['coef0']), repr(self.params['degree']))
        elif self.params['kernel'] == 'sigmoid':
            str += self.temp('kernel.sigmoid').format(
                len(self.svs), self.n_svs, repr(self.params['gamma']),
                repr(self.params['coef0']), repr(self.params['degree']))
        elif self.params['kernel'] == 'linear':
            str += self.temp('kernel.linear').format(
                len(self.svs), self.n_svs)

        # Decicion:
        n_svs = [self.temp('type').format(repr(v)) for v in self.svs_rows]
        n_svs = ', '.join(n_svs)
        str += self.temp('arr[]').format(type='int', name='n_svs', values=n_svs)
        str += self.temp('starts').format(self.n_svs_rows)
        str += self.temp('ends').format(self.n_svs_rows)
        str += self.temp('decicions').format(self.n_svs_rows)
        str += self.temp('classes').format(self.n_classes)
        str = self.indent(str, indentation=2)
        return self.temp('method', indentation=1, skipping=True).format(
            method_name=self.method_name, decicion=str)


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
