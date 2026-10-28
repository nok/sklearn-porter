from collections.abc import Callable
from copy import deepcopy
from json import dumps

import numpy as np
from loguru import logger as L
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class TransformerBase(EstimatorBase):
    """
    Base class of all ported transformers.

    A ported transformer has the method `transform`, which returns the
    transformed features of one sample. Its `main` prints the transformed
    features of each sample in one line (separated by spaces, with the shared
    template `values` of each language), so the output can be passed to a
    ported estimator and `make()` reads it without JSON.

    The model data of a transformer consists of arrays with one number per
    feature (e.g. `mean` and `scale` of a `StandardScaler`), which `port()`
    renders as arrays (`attached`), as numbers in the code (`combined`) or
    as JSON (`exported`).
    """

    DEFAULT_METHOD = enum.Method.TRANSFORM

    def __init__(self, transformer: BaseEstimator):
        super().__init__(transformer)
        # The wording of the base templates:
        self.placeholders.update(
            dict(
                kind='Transformer',
                action='a transformation',
                actions='the transformations',
            )
        )

    def _get_n_features(self, transformer: BaseEstimator) -> int:
        """
        Check that the transformer is fitted and get its number of features.

        Stateless transformers (e.g. a `Binarizer`) pass `check_is_fitted`
        without a fit, but only a fit sets the number of features.

        Parameters
        ----------
        transformer : BaseEstimator
            The transformer of scikit-learn.

        Returns
        -------
        The number of features.
        """
        try:
            check_is_fitted(transformer)
        except NotFittedError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None
        n_features = getattr(transformer, 'n_features_in_', None)
        if not isinstance(n_features, (int, np.integer)):
            raise exception.NotFittedEstimatorError(self.estimator_name)
        return int(n_features)

    def _set_model_data(self, n_features: int, **arrays: object):
        """
        Set the meta info and the model data (one number per feature).

        Parameters
        ----------
        n_features : int
            The number of features (also the number of transformed features).
        arrays : object
            The arrays of the model data with one number per feature.
        """
        self.meta_info = dict(
            n_features=n_features,
            n_features_out=n_features,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        self.model_data = {}
        for name, values in arrays.items():
            values = np.asarray(values, dtype=np.float64).ravel()
            # The literals of the languages and JSON contain no `NaN` or
            # infinite values (e.g. of a feature with missing values only):
            if not np.isfinite(values).all():
                msg = (
                    f'The model data `{name}` of the transformer '
                    f'`{self.estimator_name}` contains missing or infinite '
                    'values, which are not supported yet.'
                )
                raise exception.NotSupportedYetError(msg)
            self.model_data[name] = values.tolist()
        L.info(f'Model data (keys): {self.model_data.keys()}')
        L.opt(lazy=True).debug(f'Model data: {self.model_data}')

    def literals(self, converter: Callable[[object], str]) -> dict:
        """
        Get additional numbers (e.g. parameters) as literals for the code.

        Parameters
        ----------
        converter : Callable
            The converter of the floating numbers.

        Returns
        -------
        The placeholders of the literals.
        """
        return {}

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
        plas.update(self.literals(converter))

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Make 'exported' variant:
        if template == enum.Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        values = {
            name: list(map(converter, data))
            for name, data in self.model_data.items()
        }

        # Make 'combined' variant:
        if template == enum.Template.COMBINED:
            plas.update(values)
            tpl_class = tpls.get_template('combined.class')
            return tpl_class.render(**plas)

        # Make 'attached' variant (the names of the model data get the prefix
        # of the step of a pipeline, see `Pipeline`):
        prefix = self.template_globals.get('prefix', '')
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        for name, data in values.items():
            plas[name] = tpl_arr_1.render(
                type=tpl_double,
                name=prefix + name,
                values=', '.join(data),
                n=len(data),
            )
        tpl_class = tpls.get_template('attached.class')
        return tpl_class.render(**plas)
