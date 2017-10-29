# -*- coding: utf-8 -*-

import numpy as np
from sklearn_porter.estimator.classifier.Classifier import Classifier


class BernoulliNB(Classifier):
    """
    See also
    --------
    ...
    """

    SUPPORTED_METHODS = ['predict']

    # @formatter:off
    TEMPLATES = {
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
        }
    }
    # @formatter:on

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param estimator : AdaBoostClassifier
            An instance of a trained BernoulliNB estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(BernoulliNB, self).__init__(
            estimator, target_language=target_language,
            target_method=target_method, **kwargs)
        self.estimator = estimator

    def export(self, class_name, method_name):
        """
        Port a trained estimator to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name: string
            The name of the class in the returned result.
        :param method_name: string
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

        self.n_classes = len(est.classes_)
        self.n_features = len(est.feature_log_prob_[0])

        temp_type = self.temp('type')
        temp_arr = self.temp('arr')
        temp_arr_ = self.temp('arr[]')
        temp_arr__ = self.temp('arr[][]')

        # Create class prior probabilities:
        priors = [self.temp('type').format(self.repr(p)) for p in
                  est.class_log_prior_]
        priors = ', '.join(priors)
        self.priors = temp_arr_.format(type='double', name='priors',
                                       values=priors)

        # Create negative probabilities:
        neg_prob = np.log(1 - np.exp(est.feature_log_prob_))
        probs = []
        for prob in neg_prob:
            tmp = [temp_type.format(self.repr(p)) for p in prob]
            tmp = temp_arr.format(', '.join(tmp))
            probs.append(tmp)
        probs = ', '.join(probs)
        self.neg_probs = temp_arr__.format(type='double', name='negProbs',
                                           values=probs)

        delta_probs = (est.feature_log_prob_ - neg_prob).T
        probs = []
        for prob in delta_probs:
            tmp = [temp_type.format(self.repr(p)) for p in prob]
            tmp = temp_arr.format(', '.join(tmp))
            probs.append(tmp)
        probs = ', '.join(probs)
        self.del_probs = temp_arr__.format(type='double', name='delProbs',
                                           values=probs)

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
        Build the estimator method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        n_indents = 1 if self.target_language in ['java', 'js'] else 0
        temp_method = self.temp('method.predict', n_indents=n_indents,
                                skipping=True)
        out = temp_method.format(**self.__dict__)
        return out

    def create_class(self, method):
        """
        Build the estimator class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        self.__dict__.update(dict(method=method))
        temp_class = self.temp('class')
        out = temp_class.format(**self.__dict__)
        return out
