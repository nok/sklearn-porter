from copy import deepcopy
from json import dumps, encoder
from typing import Tuple, Union, Callable

from loguru import logger as L

# scikit-learn
from sklearn.svm import LinearSVC as LinearSVCClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class LinearSVC(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a LinearSVC classifier."""

    SKLEARN_URL = 'sklearn.svm.LinearSVC.html'

    DEFAULT_LANGUAGE = enum.Language.JAVA
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
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
        }
    }

    estimator = None  # type: LinearSVCClass

    def __init__(self, estimator: LinearSVCClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Is the estimator fitted?
        try:
            est.coef_
        except AttributeError:
            raise exception.NotFittedEstimatorError(self.estimator_name)

        self.meta_info = dict(
            n_features=len(est.coef_[0]),
            n_classes=len(est.classes_),
            is_binary=len(est.classes_) == 2
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        L.opt(lazy=True).debug('Meta info: {}'.format(self.meta_info))

        if self.meta_info['is_binary']:
            self.model_data = dict(
                coeffs=est.coef_[0].tolist(), inters=est.intercept_[0].tolist()
            )
        else:
            self.model_data = dict(
                coeffs=est.coef_.tolist(), inters=est.intercept_.tolist()
            )
        L.info('Model data (keys): {}'.format(self.model_data.keys()))
        L.opt(lazy=True).debug('Model data: {}'.format(self.model_data))

    def port(
        self,
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
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
            tpl_name = 'exported.class'
            tpl_class = tpls.get_template(tpl_name)
            out_class = tpl_class.render(**plas)
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Make 'attached' variant:
        # Pick templates:
        tpl_init = tpls.get_template('init')
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        if self.meta_info['is_binary']:

            inters_val = converter(self.model_data.get('inters'))
            inters_str = tpl_init.render(
                type=tpl_double, name='inters', value=inters_val
            )

            coeffs_val = list(map(converter, self.model_data.get('coeffs')))
            coeffs_str = tpl_arr_1.render(
                type=tpl_double,
                name='coeffs',
                values=', '.join(coeffs_val),
                n=len(coeffs_val)
            )

        else:  # is multi

            inters_val = list(map(converter, self.model_data.get('inters')))
            inters_str = tpl_arr_1.render(
                type=tpl_double,
                name='inters',
                values=', '.join(inters_val),
                n=len(inters_val)
            )

            coeffs_val = self.model_data.get('coeffs')
            coeffs_str = tpl_arr_2.render(
                type=tpl_double,
                name='coeffs',
                values=', '.join(
                    list(
                        tpl_in_brackets.render(
                            value=', '.join(list(map(converter, v)))
                        ) for v in coeffs_val
                    )
                ),
                n=len(coeffs_val),
                m=len(coeffs_val[0])
            )

        plas.update(dict(
            coeffs=coeffs_str,
            inters=inters_str,
        ))

        tpl_class = tpls.get_template('attached.class')
        out_class = tpl_class.render(**plas)
        return out_class
