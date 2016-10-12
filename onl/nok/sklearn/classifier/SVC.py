from Classifier import Classifier


class SVC(Classifier):
    """
    See also
    --------
    sklearn.svm.SVC

    http://scikit-learn.org/0.18/modules/generated/sklearn.svm.SVC.html
    """

    SUPPORT = {'predict': ['java']}

    # @formatter:off
    TEMPLATE = {
        'java': {
            'type':     ('{0}'),
            'arr':      ('{{{0}}}'),
            'arr[]':    ('\n{type}[] {name} = {{{values}}};'),
            'arr[][]':  ('\n{type}[][] {name} = {{{values}}};'),
            'kernel': {
                'linear': (  # <x,x'>
                    '\n'
                    'double[] kernels = new double[{0}]; \n'
                    'double kernel; \n'
                    'for (int i=0; i<{0}; i++) {{ \n'
                    '    kernel = 0.; \n'
                    '    for (int j=0; j<{1}; j++) {{ \n'
                    '        kernel += svs[i][j] * atts[j]; \n'
                    '    }} \n'
                    '    kernels[i] = kernel; \n'
                    '}} \n'
                ),
                'rbf': (  # exp(-y|x-x'|^2)
                    '\n'
                    'double[] kernels = new double[{0}]; \n'
                    'double kernel; \n'
                    'for (int i=0; i<{0}; i++) {{ \n'
                    '    kernel = 0.; \n'
                    '    for (int j=0; j<{1}; j++) {{ \n'
                    '        kernel += Math.pow(svs[i][j] - atts[j], 2); \n'
                    '    }} \n'
                    '    kernels[i] = Math.exp(-{2} * kernel); \n'
                    '}} \n'
                ),
                'sigmoid': (  # tanh(y<x,x'>+r)
                    '\n'
                    'double[] kernels = new double[{0}]; \n'
                    'double kernel; \n'
                    'for (int i=0; i<{0}; i++) {{ \n'
                    '    kernel = 0.; \n'
                    '    for (int j=0; j<{1}; j++) {{ \n'
                    '        kernel += svs[i][j] * atts[j]; \n'
                    '    }} \n'
                    '    kernels[i] = Math.tanh(({2} * kernel) + {3}); \n'
                    '}} \n'
                ),
                'poly': (  # (y<x,x'>+r)^d
                    '\n'
                    'double[] kernels = new double[{0}]; \n'
                    'double kernel; \n'
                    'for (int i=0; i<{0}; i++) {{ \n'
                    '    kernel = 0.; \n'
                    '    for (int j=0; j<{1}; j++) {{ \n'
                    '        kernel += svs[i][j] * atts[j]; \n'
                    '    }} \n'
                    '    kernels[i] = Math.pow(({2} * kernel) + {3}, {4}); \n'
                    '}} \n'
                )
            },
            'starts': (
                '\n'
                'int[] starts = new int[{0}]; \n'
                'for (int i=0; i<{0}; i++) {{ \n'
                '    if (i!=0) {{ \n'
                '        int start = 0;\n'
                '        for (int j=0; j<i; j++) {{ \n'
                '            start += n_svs[j]; \n'
                '        }} \n'
                '        starts[i] = start; \n'
                '    }} else {{ \n'
                '        starts[0] = 0; \n'
                '    }} \n'
                '}} \n'
            ),
            'ends': (
                '\n'
                'int[] ends = new int[{0}]; \n'
                'for (int i=0; i<{0}; i++) {{ \n'
                '    ends[i] = n_svs[i] + starts[i]; \n'
                '}} \n'
            ),
            'decicions': (
                '\n'
                'double[] decisions = new double[{0}]; \n'
                'for (int i = 0, d = 0, l = {0}; i < l; i++) {{ \n'
                '    for (int j = i + 1; j < l; j++) {{ \n'
                '        double tmp1 = 0., tmp2 = 0.; \n'
                '        for (int k = starts[j]; k < ends[j]; k++) {{ \n'
                '           tmp1 += kernels[k] * coeffs[i][k]; \n'
                '        }} \n'
                '        for (int k = starts[i]; k < ends[i]; k++) {{ \n'
                '            tmp2 += kernels[k] * coeffs[j - 1][k]; \n'
                '        }} \n'
                '        decisions[d] = tmp1 + tmp2 + inters[d++]; \n'
                '    }} \n'
                '}} \n'
            ),
            'classes': (
                '\n'
                'int[] votes = new int[{0}]; \n'
                'for (int i = 0, d = 0, l = {0}; i < l; i++) {{ \n'
                '    for (int j = i + 1; j < l; j++) {{ \n'
                '        votes[d] = decisions[d++] > 0 ? i : j; \n'
                '    }} \n'
                '}} \n'
                '\n'
                'int[] amounts = new int[{0}]; \n'
                'for (int i = 0, l = {0}; i < l; i++) {{ \n'
                '    amounts[votes[i]] += 1; \n'
                '}} \n'
                '\n'
                'int class_val = -1, class_idx = -1; \n'
                'for (int i = 0, l = {0}; i < l; i++) {{ \n'
                '    if (amounts[i] > class_val) {{ \n'
                '        class_val = amounts[i]; \n'
                '        class_idx = i; \n'
                '    }} \n'
                '}} \n'
                'return classes[class_idx]; \n'
            ),
            'method': (
                '\n'
                'public static int {0}(float[] atts) {{ \n'
                '    {1}'
                '}}'
            ),
            'class': (
                'class {0} {{ \n'
                '    {2} \n'
                '    public static void main(String[] args) {{ \n'
                '        if (args.length == {3}) {{ \n'
                '            float[] atts = new float[args.length]; \n'
                '            for (int i = 0, l = args.length; i < l; i++) {{ \n'
                '                atts[i] = Float.parseFloat(args[i]); \n'
                '            }} \n'
                '            System.out.println({0}.{1}(atts)); \n'
                '        }} \n'
                '    }} \n'
                '}}'
            )
        }
    }
    # @formatter:on


    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(SVC, self).__init__(language, method_name, class_name)


    def port(self, model):
        """Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : SVC
            An instance of a trained SVC classifier.
        """
        super(self.__class__, self).port(model)

        params = model.get_params()
        # Check kernel type:
        if params['kernel'] not in ['linear', 'rbf', 'poly', 'sigmoid']:
            msg = 'The kernel type is not supported.'
            raise ValueError(msg)
        # Check rbf gamma value:
        if params['kernel'] == 'rbf' and params['gamma'] == 'auto':
            msg = 'The classifier gamma value have to be set (currently it is \'auto\').'
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
        str = ''

        # Support vectors:
        vectors = []
        for vidx, vector in enumerate(self.svs):
            _vectors = [self.temp('type').format(repr(v)) for v in vector]
            _vectors = self.temp('arr').format(', '.join(_vectors))
            vectors.append(_vectors)
        vectors = ', '.join(vectors)
        str += self.temp('arr[][]').format(
            type='double',
            name='svs',
            values=vectors)

        # Coefficients:
        coeffs = []
        for cidx, coeff in enumerate(self.coeffs):
            _coeffs = [self.temp('type').format(repr(c)) for c in coeff]
            _coeffs = self.temp('arr').format(', '.join(_coeffs))
            coeffs.append(_coeffs)
        coeffs = ', '.join(coeffs)
        str += self.temp('arr[][]').format(
            type='double',
            name='coeffs',
            values=coeffs)

        # Interceptions:
        inters = [self.temp('type').format(repr(i)) for i in self.inters]
        inters = ', '.join(inters)
        str += self.temp('arr[]').format(
            type='double',
            name='inters',
            values=inters)

        # Classes:
        classes = [self.temp('type').format(repr(c)) for c in self.classes]
        classes = ', '.join(classes)
        str += self.temp('arr[]').format(
            type='int',
            name='classes',
            values=classes)

        # Kernels:
        if self.params['kernel'] == 'rbf':
            str += self.temp('kernel', 'rbf').format(
                len(self.svs), self.n_svs, repr(self.params['gamma']))

        if self.params['kernel'] == 'poly':
            str += self.temp('kernel', 'poly').format(
                len(self.svs), self.n_svs,
                repr(self.params['gamma']),
                repr(self.params['coef0']),
                repr(self.params['degree']))

        if self.params['kernel'] == 'sigmoid':
            str += self.temp('kernel', 'sigmoid').format(
                len(self.svs), self.n_svs,
                repr(self.params['gamma']),
                repr(self.params['coef0']),
                repr(self.params['degree']))

        if self.params['kernel'] == 'linear':
            str += self.temp('kernel', 'linear').format(
                len(self.svs),
                self.n_svs)

        # Decicion:
        n_svs = [self.temp('type').format(repr(v)) for v in self.svs_rows]
        n_svs = ', '.join(n_svs)
        str += self.temp('arr[]').format(
            type='int',
            name='n_svs',
            values=n_svs)

        str += self.temp('starts').format(self.n_svs_rows)
        str += self.temp('ends').format(self.n_svs_rows)

        str += self.temp('decicions').format(self.n_svs_rows)
        str += self.temp('classes').format(self.n_classes)

        return self.temp('method').format(self.method_name, str)


    def create_class(self, method):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        return self.temp('class').format(
            self.class_name,
            self.method_name,
            method,
            self.n_svs)
