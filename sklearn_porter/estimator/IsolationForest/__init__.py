from collections.abc import Callable

import numpy as np
from loguru import logger as L
from numpy.typing import ArrayLike

# scikit-learn
from sklearn.ensemble import IsolationForest as IsolationForestClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.RandomForestRegressor import (
    RandomForestRegressor,
)
from sklearn_porter.utils import dump_model_data, get_tree_data


def average_path_length(n_samples: ArrayLike) -> np.ndarray:
    """
    Get the average path length of an unsuccessful search in a binary
    search tree with `n_samples` samples, like scikit-learn computes it
    (`sklearn.ensemble._iforest._average_path_length`).

    Parameters
    ----------
    n_samples : array-like
        The numbers of samples, e.g. of the nodes of a tree.

    Returns
    -------
    The average path lengths.
    """
    n_samples = np.asarray(n_samples, dtype=float)
    lengths = np.zeros(n_samples.shape)
    lengths[n_samples == 2] = 1.0
    mask = n_samples > 2
    n = n_samples[mask]
    lengths[mask] = (
        2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n
    )
    return lengths


class IsolationForest(RandomForestRegressor, EstimatorBase):
    """Extract model data and port an IsolationForest outlier detector."""

    SKLEARN_URL = 'sklearn.ensemble.IsolationForest.html'

    def __init__(self, estimator: IsolationForestClass):
        EstimatorBase.__init__(self, estimator)
        L.info(f'Create specific estimator `{self.estimator_name}`.')
        est = estimator  # alias

        # Is the estimator fitted?
        try:
            est.estimators_
        except AttributeError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        n_features = est.n_features_in_  # ty: ignore[unresolved-attribute]
        self.meta_info = dict(
            n_estimators=len(est.estimators_),
            n_features=n_features,
            n_outputs=1,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        # scikit-learn sums up the path length of each tree, which is the
        # depth of the leaf of a sample plus the average path length of the
        # training samples of the leaf minus 1. The value of each leaf is this
        # path length, so the ported estimator sums up the same values in the
        # same order (`_compute_score_samples()`):
        estimators = []
        for tree, features in zip(
            est.estimators_, est.estimators_features_, strict=True
        ):
            tree_ = tree.tree_
            lengths = (
                tree_.compute_node_depths()
                + average_path_length(tree_.n_node_samples)
                - 1.0
            )
            tree_data = get_tree_data(tree_, lengths[:, np.newaxis], 'values')
            # The trees use a subset of the features with `max_features < 1`:
            if len(features) != n_features:
                indices = np.asarray(features)[tree_data['indices']]
                tree_data['indices'] = indices.tolist()
            estimators.append(tree_data)

        # The score of a sample is `-2 ** (-sum / denominator)`:
        denominator = float(
            len(est.estimators_) * average_path_length([est.max_samples_])[0]
        )
        if denominator == 0:
            # With a single training sample per tree scikit-learn uses the
            # score `-2 ** -1`, which the sum and denominator also give:
            for tree_data in estimators:
                tree_data['values'] = [1.0] * len(tree_data['values'])
            denominator = float(len(est.estimators_))

        self.model_data = dict(
            estimators=estimators,
            denominator=denominator,
            # A sample is an outlier if its score is lower than the offset:
            offset=float(est.offset_),
        )
        L.info(f'Model data (keys): {self.model_data.keys()}')
        L.opt(lazy=True).debug(f'Model data: {self.model_data}')

    def port(
        self,
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
        to_json: bool = False,
    ) -> str | tuple[str, str]:
        """
        Port an estimator.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed estimator will be
            used, e.g. `IsolationForest`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The ported estimator.
        """
        # The denominator and the offset are written as literals into the
        # source code of the templates `attached` and `combined` and are
        # part of the model data of the template `exported`:
        self.placeholders.update(
            denominator=converter(self.model_data['denominator']),
            offset=converter(self.model_data['offset']),
        )
        out = super().port(language, template, class_name, converter, to_json)
        if template == enum.Template.EXPORTED:
            out_class, _ = out
            model_data = dump_model_data(self.model_data)
            return out_class, model_data
        return out
