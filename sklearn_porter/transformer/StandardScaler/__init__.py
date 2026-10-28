from collections.abc import Callable
from copy import deepcopy
from json import dumps

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler as StandardScalerClass
from sklearn.utils.validation import check_is_fitted

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class StandardScaler(TransformerBase, EstimatorApiABC):
    """Extract model data and port a StandardScaler transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.StandardScaler.html'

    DEFAULT_LANGUAGE = enum.Language.JS
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.TRANSFORM

    SUPPORT = {
        language: {
            enum.Template.ATTACHED: {enum.Method.TRANSFORM},
            enum.Template.COMBINED: {enum.Method.TRANSFORM},
            enum.Template.EXPORTED: {enum.Method.TRANSFORM},
        }
        for language in enum.Language
    }

    def __init__(self, transformer: StandardScalerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias

        # Is the transformer fitted?
        try:
            check_is_fitted(est)
        except NotFittedError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        n_features = est.n_features_in_  # ty: ignore[unresolved-attribute]

        # scikit-learn subtracts the mean only with `with_mean=True` and
        # divides by the scale only with `with_std=True` (`mean_` or `scale_`
        # are `None` otherwise). Subtracting zero and dividing by one doesn't
        # change any value (neither `-0.0` nor `NaN`), so the ported
        # transformer always uses both:
        mean = est.mean_ if est.with_mean else np.zeros(n_features)
        scale = est.scale_ if est.with_std else np.ones(n_features)

        self.meta_info = dict(
            n_features=n_features,
            n_features_out=n_features,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        self.model_data = dict(
            mean=np.asarray(mean, dtype=np.float64).tolist(),
            scale=np.asarray(scale, dtype=np.float64).tolist(),
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
        Port a transformer.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed transformer will
            be used, e.g. `StandardScaler`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Unused, because the ported transformers always print the
            transformed features of each sample in one line.

        Returns
        -------
        The ported transformer.
        """
        # Placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(dict(class_name=class_name))
        plas.update(self.meta_info)

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Make 'exported' variant:
        if template == enum.Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        mean = list(map(converter, self.model_data['mean']))
        scale = list(map(converter, self.model_data['scale']))

        # Make 'combined' variant:
        if template == enum.Template.COMBINED:
            plas.update(dict(mean=mean, scale=scale))
            tpl_class = tpls.get_template('combined.class')
            return tpl_class.render(**plas)

        # Make 'attached' variant (the names of the model data get the prefix
        # of the step of a pipeline, see `Pipeline`):
        prefix = self.template_globals.get('prefix', '')
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        plas.update(
            dict(
                mean=tpl_arr_1.render(
                    type=tpl_double,
                    name=prefix + 'mean',
                    values=', '.join(mean),
                    n=len(mean),
                ),
                scale=tpl_arr_1.render(
                    type=tpl_double,
                    name=prefix + 'scale',
                    values=', '.join(scale),
                    n=len(scale),
                ),
            )
        )
        tpl_class = tpls.get_template('attached.class')
        return tpl_class.render(**plas)
