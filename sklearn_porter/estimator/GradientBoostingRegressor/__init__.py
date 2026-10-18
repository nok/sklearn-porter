from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier as GradientBoostingClassifierClass,
)
from sklearn.ensemble import (
    GradientBoostingRegressor as GradientBoostingRegressorClass,
)

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.RandomForestRegressor import (
    RandomForestRegressor,
)
from sklearn_porter.utils import dump_model_data, get_tree_data


def get_stages_data(
    estimator: GradientBoostingClassifierClass | GradientBoostingRegressorClass,
) -> tuple[list, list]:
    """
    Get the model data of the trees and the initial raw predictions of a
    fitted gradient boosting estimator.

    scikit-learn starts with the constant raw predictions of the initial
    estimator (`init_`) and adds the values of the leaves of all trees
    multiplied by the learning rate, stage by stage. The values of the
    leaves are multiplied by the learning rate in advance, so the ported
    estimators add exactly the same values in the same order.

    Parameters
    ----------
    estimator : GradientBoostingClassifier or GradientBoostingRegressor
        The fitted estimator.

    Returns
    -------
    The model data of the trees (stage by stage, with one tree per class
    of multiclass classifiers) and the initial raw predictions (one per
    tree of a stage).

    Raises
    ------
    NotSupportedYetError
        If the initial estimator doesn't predict constant values.
    """
    est = estimator  # alias
    init = est.init_
    # Dummy estimators predict constant values, except of the strategy
    # 'stratified', which predicts random classes:
    is_constant = isinstance(init, str) and init == 'zero'
    if isinstance(init, (DummyClassifier, DummyRegressor)):
        is_constant = init.strategy != 'stratified'
    if not is_constant:
        name = init.__class__.__qualname__
        msg = (
            f'The initial estimator `{name}` (`init`) is not supported, '
            "use the default estimator or `init='zero'` instead."
        )
        raise exception.NotSupportedYetError(msg)

    n_stages, n_trees = est.estimators_.shape
    lr = est.learning_rate
    trees = [
        get_tree_data(
            est.estimators_[i, k].tree_,
            lr * est.estimators_[i, k].tree_.value[:, :, 0],
            'values',
        )
        for i in range(n_stages)
        for k in range(n_trees)
    ]
    n_features = est.n_features_in_  # ty: ignore[unresolved-attribute]
    inits = est._raw_predict_init(np.zeros((1, n_features)))[0]
    return trees, inits.tolist()


class GradientBoostingRegressor(RandomForestRegressor, EstimatorBase):
    """Extract model data and port a GradientBoostingRegressor regressor."""

    SKLEARN_URL = 'sklearn.ensemble.GradientBoostingRegressor.html'

    def __init__(self, estimator: GradientBoostingRegressorClass):
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

        trees, inits = get_stages_data(est)
        self.meta_info = dict(
            n_estimators=len(trees),
            n_features=est.n_features_in_,  # ty: ignore[unresolved-attribute]
            n_outputs=1,  # the number of values of each leaf
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        self.model_data = dict(estimators=trees, inits=inits)
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
            used, e.g. `GradientBoostingRegressor`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The ported estimator.
        """
        # The initial raw predictions are written as literals into the
        # source code of the templates `attached` and `combined` and are
        # part of the model data of the template `exported`:
        inits = self.model_data['inits']
        self.placeholders.update(
            n_inits=len(inits),
            inits=', '.join(map(converter, inits)),
        )
        out = super().port(language, template, class_name, converter, to_json)
        if template == enum.Template.EXPORTED:
            out_class, _ = out
            return out_class, dump_model_data(self.model_data)
        return out
