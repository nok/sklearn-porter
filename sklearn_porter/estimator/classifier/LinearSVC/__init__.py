# -*- coding: utf-8 -*-

import os
import json
from json import encoder

from sklearn_porter.estimator.classifier.Classifier import Classifier


class LinearSVC(Classifier):
    """
    See also
    --------
    sklearn.svm.LinearSVC

    http://scikit-learn.org/stable/modules/generated/
    sklearn.svm.LinearSVC.html
    """

    SUPPORTED_METHODS = ['predict']

    # @formatter:off
    TEMPLATES = {
        'c': {
            'init':     '{type} {name} = {value};',
            'type':     '{0}',
            'arr':      '{{{0}}}',
            'arr[]':    '{type} {name}[{n}] = {{{values}}};',
            'arr[][]':  '{type} {name}[{n}][{m}] = {{{values}}};',
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
            'arr[]':    '{type}[] {name} = {{{values}}};',
            'arr[][]':  '{type}[][] {name} = {{{values}}};',
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

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param estimator : LinearSVC
            An instance of a trained LinearSVC estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(LinearSVC, self).__init__(estimator,
                                        target_language=target_language,
                                        target_method=target_method, **kwargs)
        self.estimator = estimator

    def export(self, class_name, method_name, export_data=False,
               export_dir='.', export_filename='data.json',
               export_append_checksum=False, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param class_name : string, default: 'Brain'
            The name of the class in the returned result.
        :param method_name : string, default: 'predict'
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

        # Estimator:
        est = self.estimator

        self.n_features = len(est.coef_[0])
        self.n_classes = len(est.classes_)
        self.is_binary = self.n_classes == 2
        self.prefix = 'binary' if self.is_binary else 'multi'

        temp_type = self.temp('type')
        temp_arr = self.temp('arr')
        temp_arr_ = self.temp('arr[]')
        temp_arr__ = self.temp('arr[][]')

        # Coefficients:
        if self.is_binary:
            coefs = est.coef_[0]
            coefs = [temp_type.format(self.repr(c)) for c in coefs]
            coefs = ', '.join(coefs)
            coefs = temp_arr_.format(type='double', name='coefficients',
                                     values=coefs, n=self.n_features)
        else:
            coefs = []
            for coef in est.coef_:
                tmp = [temp_type.format(self.repr(c)) for c in coef]
                tmp = temp_arr.format(', '.join(tmp))
                coefs.append(tmp)
            coefs = ', '.join(coefs)
            coefs = temp_arr__.format(type='double', name='coefficients',
                                      values=coefs, n=self.n_classes,
                                      m=self.n_features)
        self.coefficients = coefs

        # Intercepts:
        if self.is_binary:
            inters = est.intercept_[0]
            temp_init = self.temp('init')
            inters = temp_init.format(type='double', name='intercepts',
                                      value=self.repr(inters))
        else:
            inters = est.intercept_
            inters = [temp_type.format(self.repr(i)) for i in inters]
            inters = ', '.join(inters)
            inters = temp_arr_.format(type='double', name='intercepts',
                                      values=inters, n=self.n_classes)
        self.intercepts = inters

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
        :param temp_type : string
            The kind of export type (embedded, separated, exported).

        Returns
        -------
        :return : string
            The transpiled predict method as string.
        """
        # Exported:
        if temp_type == 'exported':
            temp = self.temp('exported.{}.class'.format(self.prefix))
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
        est = self.estimator
        coefs = est.coef_[0] if self.is_binary else est.coef_
        inters = est.intercept_[0] if self.is_binary else est.intercept_
        model_data = {
            'coefficients': coefs.tolist(),
            'intercepts': inters.tolist(),
        }
        encoder.FLOAT_REPR = lambda o: self.repr(o)
        json_data = json.dumps(model_data, sort_keys=True)
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
        n_indents = 0 if self.target_language in ['c', 'go'] else 1
        method_type = 'separated.{}.method'.format(self.prefix)
        method_temp = self.temp(method_type, n_indents=n_indents,
                                skipping=True)
        return method_temp.format(**self.__dict__)

    def create_class(self):
        """
        Build the estimator class.

        Returns
        -------
        :return : string
            The built class as string.
        """
        if self.target_language in ['java', 'go']:
            n_indents = 1 if self.target_language == 'java' else 0
            class_head_temp = self.temp('separated.{}.class'.format(
                self.prefix), n_indents=n_indents, skipping=True)
            self.class_head = class_head_temp.format(**self.__dict__)

        return self.temp('separated.class').format(**self.__dict__)
