# -*- coding: utf-8 -*-

import numpy as np
from ..Classifier import Classifier


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
    }
    # @formatter:on

    def __init__(self, model, target_language='java',
                 target_method='predict', **kwargs):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : AdaBoostClassifier
            An instance of a trained BernoulliNB model.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(BernoulliNB, self).__init__(
            model, target_language=target_language,
            target_method=target_method, **kwargs)
        self.model = model

        # self.n_features = len(model.sigma_[0])
        self.n_classes = len(model.classes_)
        self.n_features = len(model.feature_log_prob_[0])

        # jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        # jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        # Create class prior probabilities:
        priors = [self.temp('type').format(self.repr(p)) for p in
                  model.class_log_prior_]
        priors = ', '.join(priors)
        self.priors = self.temp('arr[]').format(type='double', name='priors',
                                                values=priors)

        # Create probabilities:
        # probs = []
        # for prob in model.feature_log_prob_:
        #     tmp = [self.temp('type').format(repr(p)) for p in prob]
        #     tmp = self.temp('arr').format(', '.join(tmp))
        #     probs.append(tmp)
        # probs = ', '.join(probs)
        # self.pos_probs = self.temp('arr[][]').format(type='double',
        #                                              name='posProbs',
        #                                              values=probs)

        # Create negative probabilities:
        neg_prob = np.log(1 - np.exp(model.feature_log_prob_))
        probs = []
        for prob in neg_prob:
            tmp = [self.temp('type').format(self.repr(p)) for p in prob]
            tmp = self.temp('arr').format(', '.join(tmp))
            probs.append(tmp)
        probs = ', '.join(probs)
        self.neg_probs = self.temp('arr[][]').format(type='double',
                                                     name='negProbs',
                                                     values=probs)

        delta_probs = (model.feature_log_prob_ - neg_prob).T
        probs = []
        for prob in delta_probs:
            tmp = [self.temp('type').format(self.repr(p)) for p in prob]
            tmp = self.temp('arr').format(', '.join(tmp))
            probs.append(tmp)
        probs = ', '.join(probs)
        self.del_probs = self.temp('arr[][]').format(type='double',
                                                     name='delProbs',
                                                     values=probs)

    def export(self, class_name="Brain", method_name="predict", use_repr=True):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name: string, default: 'Brain'
            The name of the class in the returned result.
        :param method_name: string, default: 'predict'
            The name of the method in the returned result.
        :param use_repr : bool, default True
            Whether to use repr() for floating-point values or not.

        Returns
        -------
        :return : string
            The transpiled algorithm with the defined placeholders.
        """
        self.class_name = class_name
        self.method_name = method_name
        self.use_repr = use_repr
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
        Build the model method or function.

        Returns
        -------
        :return out : string
            The built method as string.
        """
        n_indents = 1 if self.target_language in ['java'] else 0
        return self.temp('method.predict', n_indents=n_indents,
                         skipping=True).format(**self.__dict__)

    def create_class(self, method):
        """
        Build the model class.

        Returns
        -------
        :return out : string
            The built class as string.
        """
        self.__dict__.update(dict(method=method))
        return self.temp('class').format(**self.__dict__)
