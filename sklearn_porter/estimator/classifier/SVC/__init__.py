# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

from sklearn_porter.estimator.classifier.Classifier import Classifier


class SVC(Classifier):
    """
    See also
    --------
    sklearn.svm.SVC

    http://scikit-learn.org/stable/modules/generated/
    sklearn.svm.SVC.html
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
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param estimator : SVC
            An instance of a trained SVC estimator.
        :param target_language : string, default: 'java'
            The target programming language.
        :param target_method : string, default: 'predict'
            The target method of the estimator.
        """
        super(SVC, self).__init__(estimator, target_language=target_language,
                                  target_method=target_method, **kwargs)
        self.estimator = estimator

    def export(self, class_name, method_name, export_data=False,
               export_dir='.', export_filename='data.json',
               export_append_checksum=False, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name : string
            The name of the class in the returned result.
        :param method_name : string
            The name of the method in the returned result.
        :param export_data : bool, default: False
            Whether the model data should be saved or not.
        :param export_dir : string, default: '.' (current directory)
            The directory where the model data should be saved.
        :param export_filename : string, default: 'data.json'
            The filename of the exported model data.
        :param export_append_checksum : bool, default: False
            Whether to append the checksum to the filename or not.

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
        self.params = est.get_params()

        # Check kernel type:
        supported_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if self.params['kernel'] not in supported_kernels:
            msg = 'The kernel type is not supported.'
            raise ValueError(msg)

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
        self.kernel = str(self.params['kernel'])
        if self.target_language == 'c':
            self.kernel = self.kernel[0]

        # Gamma:
        self.gamma = self.params['gamma']
        if self.gamma == 'auto':
            self.gamma = 1. / self.n_features
        self.gamma = self.repr(self.gamma)

        # Coefficient and degree:
        self.coef0 = self.repr(self.params['coef0'])
        self.degree = self.repr(self.params['degree'])

        if self.target_method == 'predict':
            # Exported:
            if export_data and os.path.isdir(export_dir):
                self.export_data(export_dir, export_filename,
                                 export_append_checksum)
                return self.predict('exported')
            # Separated:
            return self.predict('separated')

    def predict(self, temp_type):
        """
        Transpile the predict method.

        Parameters
        ----------
        :param temp_type: string
            The kind of export type (embedded, separated, exported).

        Returns
        -------
        :return : string
            The transpiled predict method as string.
        """
        # Exported:
        if temp_type == 'exported':
            temp = self.temp('exported.class')
            return temp.format(class_name=self.class_name,
                               method_name=self.method_name)
        # Separated
        self.method = self.create_method()
        return self.create_class()

    def export_data(self, directory, filename, with_md5_hash=False):
        """
        Save model data in a JSON file.

        Parameters
        ----------
        :param directory : string
            The directory.
        :param filename : string
            The filename.
        :param with_md5_hash : bool
            Whether to append the checksum to the filename or not.
        """
        model_data = {
            'vectors': self.estimator.support_vectors_.tolist(),
            'coefficients': self.estimator.dual_coef_.tolist(),
            'intercepts': self.estimator._intercept_.tolist(),
            'weights': self.estimator.n_support_.tolist(),
            'kernel': self.kernel,
            'gamma': float(self.gamma),
            'coef0': float(self.coef0),
            'degree': float(self.degree),
            'nClasses': int(self.n_classes),
            'nRows': int(self.n_svs_rows)
        }
        encoder.FLOAT_REPR = lambda o: self.repr(o)
        json_data = dumps(model_data, sort_keys=True)
        if with_md5_hash:
            import hashlib
            json_hash = hashlib.md5(json_data).hexdigest()
            filename = filename.split('.json')[0] + '_' + json_hash + '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as fp:
            fp.write(json_data)

    def create_method(self):
        """
        Build the estimator method or function.

        Returns
        -------
        :return : string
            The built method as string.
        """
        n_indents = 1 if self.target_language in ['java', 'js',
                                                  'php', 'ruby'] else 0
        return self.temp('separated.method', n_indents=n_indents,
                         skipping=True).format(**self.__dict__)

    def create_class(self):
        """
        Build the estimator class.

        Returns
        -------
        :return : string
            The built class as string.
        """
        temp_class = self.temp('separated.class')
        return temp_class.format(**self.__dict__)
