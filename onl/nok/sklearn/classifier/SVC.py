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
        if params['kernel'] != 'linear' and params['kernel'] != 'rbf':
            msg = 'The kernel type is not supported.'
            raise ValueError(msg)
        # Check rbf gamma value:
        if params['kernel'] == 'rbf' and params['gamma'] == 'auto':
            msg = 'The classifier gamma value have to be set (currently it is \'auto\').'
            raise ValueError(msg)
        self.params = params

        self.svs = model.support_vectors_
        self.n_svs = model.n_support_
        self.n_vs = len(model.support_vectors_[0])
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
        # str_class = self.create_class()
        str_method = self.create_method()
        return str_method

        # out = str_class.format(str_method)
        # return out


    def create_method(self):
        """Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """

        out = ''
        if self.language is 'java':

            # Support vectors:
            vectors = []
            for vidx, vector in enumerate(self.svs):
                _vectors = ['{0}f'.format(v.astype('|S32')) for v in vector]
                _vectors = '{{ {0} }}'.format(', '.join(_vectors))
                vectors.append(_vectors)
            vectors = 'float[][] svs = {{ {0} }};'.format(', '.join(vectors))
            out += vectors

            # Kernels:
            if self.params['kernel'] is 'rbf':
                template = {
                    'java': (
                        'float[] kernels = new float[{0}]; \n'
                        'float kernel; \n'
                        'for (int i=0; i<{0}; i++) {{ \n'
                        '    kernel = 0.0f; \n'
                        '    for (int j=0; j<{1}; j++) {{ \n'
                        '        float delta = svs[i][j] - atts[j]; \n'
                        '        kernel += delta * delta; \n'
                        '    }} \n'
                        '    kernels[i] = Math.exp(-{2}f * kernel); \n'
                        '}} \n'
                    )
                }
                out += '\n' + template[self.language].format(
                    len(self.svs), self.n_vs, self.params['gamma'])

                n_svs = ['{0}'.format(v.astype('|S32')) for v in self.n_svs]
                out += '\n' + 'int[] n_svs = {{ {0} }};'.format(', '.join(n_svs))

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
                out += '\n' + template[self.language].format(len(self.n_svs))

                template = {
                    'java': (
                        'int[] ends = new int[{0}]; \n'
                        'for (int i=0; i<{0}; i++) {{ \n'
                        '    ends[i] = n_svs[i] + starts[i]; \n'
                        '}} \n'
                    )
                }
                out += '\n' + template[self.language].format(len(self.n_svs))

                # template = {
                #     'java': (
                #         'float[] c = new float[{0}]; \n'
                #         'for (int i=0; i<{0}; i++) {{ \n'
                #         '    for (int j=1; j<{0}; j++) {{ \n'
                #         '        float a, b; \n'
                #         '        for (int j=0; j<{1}; j++) {{ \n'
                #         '            float delta = svs[i][j] - atts[j]; \n'
                #         '            kernel += delta * delta; \n'
                #         '        }} \n'
                #         '        kernels[i] = Math.exp(-{2}f * kernel); \n'
                #         '    }} \n'
                #         '}} \n'
                #     )
                # }

                # c = []
                # for i in range(len(nv)):
                #     for j in range(i + 1, len(nv)):
                #         o_a = 0.0
                #         for p in range(starts[j], ends[j]):
                #             o_a += a[i][p] * kernels[p]
                #         o_b = 0.0
                #         for p in range(starts[i], ends[i]):
                #             o_b += a[j - 1][p] * kernels[p]
                #         c.append(o_a + o_b)

                # starts = []
                # for i in range(len(nv)):
                #     if i != 0:
                #         start = 0
                #         for j in range(i):
                #             start += nv[j]
                #         starts.append(start)
                #     else:
                #         starts.append(0)
                # print(starts)
                # ends = []
                # for i in range(len(nv)):
                #     ends.append(nv[i] + starts[i])
                # print(ends)

            # Coefficients:
            # coefs = []
            # for idx, coef in enumerate(self.svs):
            #     _coefs = ['{0}f'.format(str(c)) for c in coef]
            #     _vector = '{{ {0} }}'.format(', '.join(_coefs))
            #     coefs.append(_coefs)
            # coefs = 'float[][] coefs = {{ {0} }};'.format(', '.join(coefs))
            #
            # # Interceptions:
            # inters = ['{0}f'.format(str(interc)) for interc in self.model.intercept_]
            # inters = 'float[] inters = {{ {0} }};'.format(', '.join(inters))
        #
        # template = {
        #     'java': (
        #         # @formatter:off
        #         'public static int {0}(float[] atts) {{ \n'
        #         '    if (atts.length != {1}) {{ return -1; }} \n'
        #         '    {3} \n'
        #         '    {4} \n'
        #         '    float[] classes = new float[{2}]; \n'
        #         '    for (int i = 0; i < {2}; i++) {{ \n'
        #         '        float prob = 0.0f; \n'
        #         '        for (int j = 0; j < {1}; j++) {{ \n'
        #         '            prob += coefs[i][j] * atts[j]; \n'
        #         '        }} \n'
        #         '        classes[i] = prob + inters[i]; \n'
        #         '    }} \n'
        #         '    int idx = 0; \n'
        #         '    float val = classes[0]; \n'
        #         '    for (int i = 1; i < {2}; i++) {{ \n'
        #         '        if (classes[i] > val) {{ \n'
        #         '            idx = i; \n'
        #         '            val = classes[i]; \n'
        #         '        }} \n'
        #         '    }} \n'
        #         '    return idx; \n'
        #         '}}'
        #         # @formatter:on
        #     )
        # }
        #
        # out = template[self.language].format(
        #     self.method_name,
        #     self.n_features,
        #     self.n_classes,
        #     coefs,
        #     inters)
        #
        return out


    def create_class(self):
        """Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        template = {
            'java': (
                # @formatter:off
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
                # @formatter:on
            )
        }

        out = template[self.language].format(
            self.class_name,
            self.n_features)

        return out
