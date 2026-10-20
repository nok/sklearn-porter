from collections.abc import Callable
from copy import deepcopy
from json import dumps

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.neighbors import KNeighborsRegressor as KNeighborsRegressorClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.KNeighborsClassifier import get_metric


class KNeighborsRegressor(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a KNeighborsRegressor regressor."""

    SKLEARN_URL = 'sklearn.neighbors.KNeighborsRegressor.html'

    DEFAULT_LANGUAGE = enum.Language.JAVA
    DEFAULT_TEMPLATE = enum.Template.EXPORTED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.CS: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.GO: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
    }

    def __init__(self, estimator: KNeighborsRegressorClass):
        super().__init__(estimator)
        L.info(f'Create specific estimator `{self.estimator_name}`.')
        est = estimator  # alias

        # Is the estimator fitted?
        try:
            est.n_samples_fit_
        except AttributeError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        metric, power = get_metric(est)

        # The targets of a regressor with multiple outputs have one column
        # per output. The targets of all outputs of a sample are stored one
        # after another:
        y = np.asarray(est._y, dtype=float)
        n_outputs = 1 if y.ndim == 1 else y.shape[1]

        self.meta_info = dict(
            n_outputs=n_outputs,
            is_multi_output=y.ndim > 1,
            n_templates=len(est._fit_X),
            n_features=len(est._fit_X[0]),
            metric=metric,
            weights=est.weights,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        self.model_data = dict(
            X=est._fit_X.tolist(),
            y=y.ravel().tolist(),
            k=est.n_neighbors,  # number of relevant neighbors
            metric=metric,  # `minkowski`, `chebyshev`, `cosine` or `hamming`
            power=power,  # only used by `minkowski`
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
            used, e.g. `KNeighborsRegressor`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The ported estimator.
        """
        # Placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(
            dict(
                class_name=class_name,
                to_json=to_json,
            )
        )
        plas.update(self.meta_info)

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Make 'exported' variant:
        if template == enum.Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Make 'attached' variant:
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        x_val = self.model_data['X']
        x_str = tpl_arr_2.render(
            type=tpl_double,
            name='X',
            values=', '.join(
                tpl_in_brackets.render(value=', '.join(map(converter, v)))
                for v in x_val
            ),
            n=len(x_val),
            m=len(x_val[0]),
        )
        y_val = list(map(converter, self.model_data['y']))
        y_str = tpl_arr_1.render(
            type=tpl_double, name='y', values=', '.join(y_val), n=len(y_val)
        )

        plas.update(
            dict(
                X=x_str,
                y=y_str,
                k=self.model_data['k'],
                power=self.model_data['power'],
            )
        )
        tpl_class = tpls.get_template('attached.class')
        return tpl_class.render(**plas)
