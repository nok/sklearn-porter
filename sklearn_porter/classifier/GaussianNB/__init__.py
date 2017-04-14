# -*- coding: utf-8 -*-

from ...Template import Template


class GaussianNB(Template):
    """
    See also
    --------
    sklearn.naive_bayes.BernoulliNB

    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
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
            An instance of a trained GaussianNB model.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(GaussianNB, self).__init__(
            model, target_language=target_language,
            target_method=target_method, **kwargs)
        self.model = model

        self.n_features = len(model.sigma_[0])
        self.n_classes = len(model.classes_)

        # Create class prior probabilities:
        priors = [self.temp('type').format(repr(c)) for c in
                  self.model.class_prior_]
        priors = ', '.join(priors)
        self.priors = self.temp('arr[]').format(type='double', name='priors',
                                                values=priors)

        # Create sigmas:
        sigmas = []
        for sigma in self.model.sigma_:
            tmp = [self.temp('type').format(repr(s)) for s in sigma]
            tmp = self.temp('arr').format(', '.join(tmp))
            sigmas.append(tmp)
        sigmas = ', '.join(sigmas)
        self.sigmas = self.temp('arr[][]').format(type='double', name='sigmas',
                                                  values=sigmas)

        # Create thetas:
        thetas = []
        for theta in self.model.theta_:
            tmp = [self.temp('type').format(repr(t)) for t in theta]
            tmp = self.temp('arr').format(', '.join(tmp))
            thetas.append(tmp)
        thetas = ', '.join(thetas)
        self.thetas = self.temp('arr[][]').format(type='double', name='thetas',
                                                  values=thetas)

    def export(self, class_name, method_name):
        """
        Port a trained model to the syntax of a chosen programming language.

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
        self.class_name = class_name
        self.method_name = method_name
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
