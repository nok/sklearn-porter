from collections.abc import Callable
from copy import deepcopy
from json import dumps

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.linear_model import LogisticRegression as LogisticRegressionClass
from sklearn.linear_model import Perceptron as PerceptronClass
from sklearn.linear_model import RidgeClassifier as RidgeClassifierClass
from sklearn.linear_model import SGDClassifier as SGDClassifierClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.utils import is_multilabel


class LogisticRegression(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a LogisticRegression classifier."""

    SKLEARN_URL = 'sklearn.linear_model.LogisticRegression.html'

    DEFAULT_LANGUAGE = enum.Language.JS
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.GO: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
    }

    def __init__(
        self,
        estimator: LogisticRegressionClass
        | SGDClassifierClass
        | RidgeClassifierClass
        | PerceptronClass,
    ):
        super().__init__(estimator)
        L.info(f'Create specific estimator `{self.estimator_name}`.')
        est = estimator  # alias

        # Is the estimator fitted?
        try:
            est.coef_
        except AttributeError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        # Binary classifiers have one row of coefficients, multiclass
        # classifiers one row per class. Some linear classifiers store a
        # single row as vector and the intercept as scalar:
        coeffs = np.atleast_2d(est.coef_)  # ty: ignore[no-matching-overload]
        inters = np.broadcast_to(  # ty: ignore[no-matching-overload]
            est.intercept_, (len(coeffs),)
        )

        self.meta_info = dict(
            n_features=coeffs.shape[1],
            n_classes=len(est.classes_),  # ty: ignore[unresolved-attribute]
            n_decisions=len(coeffs),
            is_binary=len(coeffs) == 1,
            # Multilabel classifiers (e.g. a RidgeClassifier fitted on a label
            # indicator matrix) predict one binary value per label:
            is_multilabel=is_multilabel(est),
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        self.model_data = dict(
            coeffs=coeffs.tolist(),
            inters=inters.tolist(),
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
            used, e.g. `LogisticRegression`.
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
                # Linear classifiers without probabilities share the
                # templates, but don't render the method `predict_proba`:
                with_proba=enum.Method.PREDICT_PROBA
                in self.SUPPORT[language][template],
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

        coeffs = [list(map(converter, c)) for c in self.model_data['coeffs']]
        inters = list(map(converter, self.model_data['inters']))

        # Make 'combined' variant:
        if template == enum.Template.COMBINED:
            plas.update(dict(coeffs=coeffs, inters=inters))
            tpl_class = tpls.get_template('combined.class')
            return tpl_class.render(**plas)

        # Make 'attached' variant:
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        coeffs_str = tpl_arr_2.render(
            type=tpl_double,
            name='coeffs',
            values=', '.join(
                tpl_in_brackets.render(value=', '.join(c)) for c in coeffs
            ),
            n=len(coeffs),
            m=len(coeffs[0]),
        )
        inters_str = tpl_arr_1.render(
            type=tpl_double,
            name='inters',
            values=', '.join(inters),
            n=len(inters),
        )
        plas.update(dict(coeffs=coeffs_str, inters=inters_str))

        tpl_class = tpls.get_template('attached.class')
        return tpl_class.render(**plas)
