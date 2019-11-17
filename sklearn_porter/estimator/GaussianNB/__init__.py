# -*- coding: utf-8 -*-

from copy import deepcopy
from json import dumps, encoder
from typing import Optional, Tuple, Union, Callable

from loguru import logger as L

# scikit-learn
from sklearn.naive_bayes import GaussianNB as GaussianNBClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class GaussianNB(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a GaussianNB classifier."""

    DEFAULT_LANGUAGE = enum.Language.JAVA
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.JAVA: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
    }

    estimator = None  # type: GaussianNBClass

    def __init__(self, estimator: GaussianNBClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Is the estimator fitted?
        try:
            est.sigma_
        except AttributeError:
            raise exception.NotFittedEstimatorError(self.estimator_name)

        self.meta_info = dict(
            n_features=len(est.sigma_[0]),
            n_classes=len(est.classes_),
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        L.opt(lazy=True).debug('Meta info: {}'.format(self.meta_info))

        self.model_data = dict(
            priors=est.class_prior_.tolist(),
            sigmas=est.sigma_.tolist(),
            thetas=est.theta_.tolist(),
        )
        L.info('Model data (keys): {}'.format(self.model_data.keys()))
        L.opt(lazy=True).debug('Model data: {}'.format(self.model_data))

    def port(
        self,
        language: Optional[enum.Language],
        template: Optional[enum.Template],
        class_name: Optional[str],
        converter: Optional[Callable[[object], str]],
        to_json: bool = False,
    ) -> Union[str, Tuple[str, str]]:
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
        plas.update(dict(
            class_name=class_name,
            to_json=to_json,
        ))
        plas.update(self.meta_info)

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Make 'exported' variant:
        if template == enum.Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Make 'attached' variant:
        # Pick templates:
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        priors_val = self.model_data.get('priors')
        priors_val_converted = list(map(converter, priors_val))
        priors_str = tpl_arr_1.render(
            type=tpl_double,
            name='priors',
            values=', '.join(priors_val_converted),
            n=len(priors_val_converted)
        )

        sigmas_val = self.model_data.get('sigmas')
        sigmas_str = tpl_arr_2.render(
            type=tpl_double,
            name='sigmas',
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    ) for v in sigmas_val
                )
            ),
            n=len(sigmas_val),
            m=len(sigmas_val[0])
        )

        thetas_val = self.model_data.get('thetas')
        thetas_str = tpl_arr_2.render(
            type=tpl_double,
            name='thetas',
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    ) for v in thetas_val
                )
            ),
            n=len(thetas_val),
            m=len(thetas_val[0])
        )

        plas.update(
            dict(
                priors=priors_str,
                sigmas=sigmas_str,
                thetas=thetas_str,
            )
        )

        tpl_class = tpls.get_template('attached.class')
        out_class = tpl_class.render(**plas)

        return out_class
