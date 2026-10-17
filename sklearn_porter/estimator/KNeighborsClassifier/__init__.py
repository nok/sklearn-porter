from collections.abc import Callable
from copy import deepcopy
from json import dumps

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifierClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.utils import get_labels, get_output_classes


class KNeighborsClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a KNeighborsClassifier classifier."""

    SKLEARN_URL = 'sklearn.neighbors.KNeighborsClassifier.html'

    DEFAULT_LANGUAGE = enum.Language.JAVA
    DEFAULT_TEMPLATE = enum.Template.EXPORTED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.GO: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
    }

    def __init__(self, estimator: KNeighborsClassifierClass):
        super().__init__(estimator)
        L.info(f'Create specific estimator `{self.estimator_name}`.')
        est = estimator  # alias

        # Is the estimator fitted?
        try:
            est.classes_
        except AttributeError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        # The votes of the neighbors are equal (`uniform`) or weighted by the
        # inverse of their distances (`distance`), callables aren't supported:
        if est.weights not in ('uniform', 'distance'):
            msg = 'Only the weights `uniform` and `distance` are supported.'
            raise exception.NotSupportedYetError(msg)

        # The effective metric and its parameters contain the used distance,
        # e.g. `manhattan` for `p=1` or `{'p': 3}` from `metric_params`. The
        # ported estimators support the metrics `minkowski`, `chebyshev`,
        # `cosine` and `hamming`. The power is only used by the Minkowski
        # distance, the other metrics keep the default power (`p=2`):
        metric = est.effective_metric_
        metric_params = est.effective_metric_params_
        power = 2
        if metric in ('euclidean', 'l2'):
            metric = 'minkowski'
        elif metric in ('manhattan', 'cityblock', 'l1'):
            metric, power = 'minkowski', 1
        elif metric == 'minkowski' and np.isfinite(metric_params['p']):
            power = metric_params['p']
        elif metric in ('minkowski', 'infinity'):  # e.g. `minkowski`, `p=inf`
            metric = 'chebyshev'
        elif metric not in ('chebyshev', 'cosine', 'hamming'):
            name = getattr(metric, '__name__', metric)
            msg = (
                f'The metric `{name}` is not supported, only the metrics '
                '`minkowski` (e.g. `euclidean` or `manhattan`), `chebyshev`, '
                '`cosine` and `hamming`.'
            )
            raise exception.NotSupportedYetError(msg)
        if metric_params.get('w') is not None:
            msg = 'The weighted Minkowski distance (`w`) is not supported.'
            raise exception.NotSupportedYetError(msg)

        # The class votes of all outputs are concatenated, `output_classes`
        # defines their segments:
        output_classes = get_output_classes(est)
        # The labels of the classes, which the ported classifier returns:
        labels, label_kind = get_labels(est)
        self.meta_info = dict(
            n_classes=sum(output_classes),
            n_outputs=len(output_classes),
            output_classes=output_classes,
            is_multi_output=len(output_classes) > 1,
            n_templates=len(est._fit_X),
            n_features=len(est._fit_X[0]),
            metric=metric,
            weights=est.weights,
            labels=labels,
            label_kind=label_kind,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        # The classes of multiple outputs are concatenated, so the indices of
        # the classes of all outputs of a sample are stored one after another:
        y = est._y.astype(int)  # ty: ignore[unresolved-attribute]
        if y.ndim > 1:
            offsets = np.cumsum([0] + output_classes[:-1])
            y = (y + offsets).ravel()

        self.model_data = dict(
            X=est._fit_X.tolist(),
            y=y.tolist(),
            k=est.n_neighbors,  # number of relevant neighbors
            n=sum(output_classes),  # number of classes (of all outputs)
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
            used, e.g. `DecisionTreeClassifier`.
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

        # Pick templates:
        tpl_int = tpls.get_template('int').render()
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        # Make 'attached' variant:
        x_val = self.model_data['X']
        x_str = tpl_arr_2.render(
            type=tpl_double,
            name='X',
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    )
                    for v in x_val
                )
            ),
            n=len(x_val),
            m=len(x_val[0]),
        )

        y_val = list(map(str, self.model_data['y']))
        y_str = tpl_arr_1.render(
            type=tpl_int, name='y', values=', '.join(y_val), n=len(y_val)
        )

        tpl_class = tpls.get_template('attached.class')
        plas.update(
            dict(
                X=x_str,
                y=y_str,
                k=self.model_data['k'],
                n=self.model_data['n'],
                power=self.model_data['power'],
            )
        )
        out_class = tpl_class.render(**plas)
        return out_class
