import numpy as np

# scikit-learn
from sklearn.naive_bayes import ComplementNB as ComplementNBClass
from sklearn.naive_bayes import MultinomialNB as MultinomialNBClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.MultinomialNB import MultinomialNB


class ComplementNB(MultinomialNB, EstimatorBase):
    """Extract model data and port a ComplementNB classifier."""

    SKLEARN_URL = 'sklearn.naive_bayes.ComplementNB.html'

    def __init__(self, estimator: ComplementNBClass):
        # The weights of the features (`feature_log_prob_`) are the negative
        # (or normalized with `norm=True`) log probabilities of the complement
        # classes, so the most likely class has the highest joint log
        # likelihood like with MultinomialNB:
        super().__init__(estimator)

    @staticmethod
    def _get_priors(
        estimator: MultinomialNBClass | ComplementNBClass,
    ) -> np.ndarray:
        """
        Get the log priors of the classes, which are added to the joint log
        likelihoods. Like scikit-learn, ComplementNB uses the priors only with
        a single class, otherwise `class_prior` and `fit_prior` are ignored.

        Parameters
        ----------
        estimator : MultinomialNB or ComplementNB
            The fitted estimator.

        Returns
        -------
        priors : np.ndarray
            The log priors of the classes.
        """
        priors = np.asarray(estimator.class_log_prior_, dtype=float)
        if len(estimator.classes_) == 1:
            return priors
        return np.zeros_like(priors)
