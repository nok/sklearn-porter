# -*- coding: utf-8 -*-

import numpy as np

from ...Model import Model


class BernoulliNB(Model):
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

    def __init__(self, language='java', method_name='predict',
                 class_name='Tmp'):
        super(BernoulliNB, self).__init__(language, method_name, class_name)

    def port(self, model):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : GaussianNB
            An instance of a trained GaussianNB classifier.
        """
        super(BernoulliNB, self).port(model)

        # self.n_features = len(model.sigma_[0])
        self.n_classes = len(model.classes_)
        self.n_features = len(model.feature_log_prob_[0])

        # jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        # jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        # Create class prior probabilities:
        priors = [self.temp('type').format(repr(p)) for p in
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
            tmp = [self.temp('type').format(repr(p)) for p in prob]
            tmp = self.temp('arr').format(', '.join(tmp))
            probs.append(tmp)
        probs = ', '.join(probs)
        self.neg_probs = self.temp('arr[][]').format(type='double',
                                                     name='negProbs',
                                                     values=probs)

        delta_probs = (model.feature_log_prob_ - neg_prob).T
        probs = []
        for prob in delta_probs:
            tmp = [self.temp('type').format(repr(p)) for p in prob]
            tmp = self.temp('arr').format(', '.join(tmp))
            probs.append(tmp)
        probs = ', '.join(probs)
        self.del_probs = self.temp('arr[][]').format(type='double',
                                                     name='delProbs',
                                                     values=probs)

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
        indent = 1 if self.language in ['java'] else 0
        return self.temp('method.predict', n_indents=indent,
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
