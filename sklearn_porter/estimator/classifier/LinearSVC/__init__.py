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

    http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
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
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param estimator : LinearSVC
            An instance of a trained AdaBoostClassifier estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(LinearSVC, self).__init__(estimator, target_language=target_language,
                                        target_method=target_method, **kwargs)
        self.estimator = estimator

    def export(self, class_name, method_name,
               export_data=False, export_dir='.',
               **kwargs):
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
                self.export_data(export_dir)
                return self.predict('exported')
            # Separated:
            return self.predict('separated')

    def predict(self, temp_type):
        """
        Transpile the predict method.

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

    def export_data(self, export_dir):
        model_data = {
            'coefficients': (self.estimator.coef_[0] if self.is_binary else self.estimator.coef_).tolist(),
            'intercepts': (self.estimator.intercept_[0] if self.is_binary else self.estimator.intercept_).tolist(),
        }
        encoder.FLOAT_REPR = lambda o: self.repr(o)
        path = os.path.join(export_dir, 'data.json')
        with open(path, 'w') as fp:
            json.dump(model_data, fp)

    def create_method(self):
        """
        Build the estimator method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        n_indents = 0 if self.target_language in ['c', 'go'] else 1
        method_type = 'separated.{}.method'.format(self.prefix)
        method_temp = self.temp(method_type, n_indents=n_indents, skipping=True)
        output = method_temp.format(**self.__dict__)
        return output

    def create_class(self):
        """
        Build the estimator class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        if self.target_language in ['java', 'go']:
            n_indents = 1 if self.target_language == 'java' else 0
            class_head_temp = self.temp('separated.{}.class'.format(self.prefix),
                                        n_indents=n_indents, skipping=True)
            self.class_head = class_head_temp.format(**self.__dict__)

        output = self.temp('separated.class').format(**self.__dict__)
        return output
