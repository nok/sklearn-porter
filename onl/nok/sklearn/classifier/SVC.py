from Classifier import Classifier


class SVC(Classifier):
    """
    See also
    --------
    sklearn.svm.SVC

    http://scikit-learn.org/0.17/modules/generated/sklearn.svm.SVC.html
    """

    SUPPORT = {
        'predict': ['java']
    }

    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        super(self.__class__, self).__init__(language, method_name, class_name)


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
        str_class = self.create_class()
        str_method = self.create_method()
        out = str_class.format(str_method)
        return out


    def create_method(self):
        """Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """

        out = ''

        # Support vectors:
        vectors = []
        for vidx, vector in enumerate(self.svs):
            _vectors = ['{0}'.format(v.astype('|S')) for v in vector]
            # _vectors = ['{0}'.format(repr(v)) for v in vector]
            _vectors = '{{ {0} }}'.format(', '.join(_vectors))
            vectors.append(_vectors)
        vectors = 'double[][] svs = {{ {0} }};'.format(', '.join(vectors))
        out += vectors

        # Coefficients:
        coeffs = []
        for cidx, coeff in enumerate(self.coeffs):
            _coeffs = ['{0}'.format(c.astype('|S')) for c in coeff]
            # _coeffs = ['{0}'.format(repr(c)) for c in coeff]
            _coeffs = '{{ {0} }}'.format(', '.join(_coeffs))
            coeffs.append(_coeffs)
        out += '\n' + 'double[][] coeffs = {{ {0} }};'.format(', '.join(coeffs))

        # Interceptions:
        # inters = ['{0}'.format(i.astype('|S')) for i in self.inters]
        inters = ['{0}'.format(repr(i)) for i in self.inters]
        out += '\n' + 'double[] inters = {{ {0} }};'.format(', '.join(inters))

        # Classes:
        classes = ['{0}'.format(c.astype('|S')) for c in self.classes]
        # classes = ['{0}'.format(repr(c)) for c in self.classes]
        out += '\n' + 'int[] classes = {{ {0} }};'.format(', '.join(classes)) + '\n'

        # Kernels:
        if self.params['kernel'] == 'rbf':
            # exp(-y|x-x'|^2)
            # @formatter:off
            template = {
                'java': (
                    'double[] kernels = new double[{0}]; \n'
                    'double kernel; \n'
                    'for (int i=0; i<{0}; i++) {{ \n'
                    '    kernel = 0.; \n'
                    '    for (int j=0; j<{1}; j++) {{ \n'
                    '        kernel += Math.pow(svs[i][j] - atts[j], 2); \n'
                    '    }} \n'
                    '    kernels[i] = Math.exp(-{2} * kernel); \n'
                    '}} \n'
                )
            }
            # @formatter:on
            out += '\n' + template[self.language].format(
                len(self.svs), self.n_svs, repr(self.params['gamma']))

        if self.params['kernel'] == 'poly':
            # (y<x,x'>+r)^d
            # @formatter:off
            template = {
                'java': (
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
            }
            # @formatter:on
            out += '\n' + template[self.language].format(
                len(self.svs), self.n_svs,
                repr(self.params['gamma']),
                repr(self.params['coef0']),
                repr(self.params['degree']))

        if self.params['kernel'] == 'sigmoid':
            # tanh(y<x,x'>+r)
            # @formatter:off
            template = {
                'java': (
                    'double[] kernels = new double[{0}]; \n'
                    'double kernel; \n'
                    'for (int i=0; i<{0}; i++) {{ \n'
                    '    kernel = 0.; \n'
                    '    for (int j=0; j<{1}; j++) {{ \n'
                    '        kernel += svs[i][j] * atts[j]; \n'
                    '    }} \n'
                    '    kernels[i] = Math.tanh(({2} * kernel) + {3}); \n'
                    '}} \n'
                )
            }
            # @formatter:on
            out += '\n' + template[self.language].format(
                len(self.svs), self.n_svs,
                repr(self.params['gamma']),
                repr(self.params['coef0']),
                repr(self.params['degree']))

        if self.params['kernel'] == 'linear':
            # <x,x'>
            # @formatter:off
            template = {
                'java': (
                    'double[] kernels = new double[{0}]; \n'
                    'double kernel; \n'
                    'for (int i=0; i<{0}; i++) {{ \n'
                    '    kernel = 0.; \n'
                    '    for (int j=0; j<{1}; j++) {{ \n'
                    '        kernel += svs[i][j] * atts[j]; \n'
                    '    }} \n'
                    '    kernels[i] = kernel; \n'
                    '}} \n'
                )
            }
            # @formatter:on
            out += '\n' + template[self.language].format(
                len(self.svs), self.n_svs)

        n_svs = ['{0}'.format(v.astype('|S')) for v in self.svs_rows]
        # n_svs = ['{0}'.format(repr(v)) for v in self.svs_rows]
        out += '\n' + 'int[] n_svs = {{ {0} }};'.format(', '.join(n_svs))

        # @formatter:off
        template = {
            'java': (
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
            )
        }
        # @formatter:on
        out += '\n' + template[self.language].format(self.n_svs_rows)

        # @formatter:off
        template = {
            'java': (
                'int[] ends = new int[{0}]; \n'
                'for (int i=0; i<{0}; i++) {{ \n'
                '    ends[i] = n_svs[i] + starts[i]; \n'
                '}} \n'
            )
        }
        # @formatter:on
        out += '\n' + template[self.language].format(self.n_svs_rows)

        # @formatter:off
        template = {
            'java': (
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
            )
        }
        # @formatter:on
        out += '\n' + template[self.language].format(self.n_svs_rows)

        # @formatter:off
        template = {
            'java': (
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
            )
        }
        # @formatter:on
        out += '\n' + template[self.language].format(self.n_classes)

        # @formatter:off
        template = {
            'java': (
                'public static int {0}(float[] atts) {{ \n'
                '    {1}'
                '}}'
            )
        }
        # @formatter:on
        out = template[self.language].format(self.method_name, out)

        return out


    def create_class(self):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        # @formatter:off
        template = {
            'java': (
                'class {0} {{{{ \n'
                '    {{0}} \n'
                '    public static void main(String[] args) {{{{ \n'
                '        if (args.length == {1}) {{{{ \n'
                '            float[] atts = new float[args.length]; \n'
                '            for (int i = 0, l = args.length; i < l; i++) {{{{ \n'
                '                atts[i] = Float.parseFloat(args[i]); \n'
                '            }}}} \n'
                '            System.out.println({0}.predict(atts)); \n'
                '        }}}} \n'
                '    }}}} \n'
                '}}}}'
            )
        }
        # @formatter:on

        out = template[self.language].format(
            self.class_name,
            self.n_svs)

        return out
