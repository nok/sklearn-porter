from copy import deepcopy
from json import dumps, encoder
from typing import Tuple, Union, Callable

from loguru import logger as L

# scikit-learn
from sklearn.neighbors import \
    KNeighborsClassifier as KNeighborsClassifierClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class KNeighborsClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a KNeighborsClassifier classifier."""

    SKLEARN_URL = 'sklearn.neighbors.KNeighborsClassifier.html'

    DEFAULT_LANGUAGE = enum.Language.JAVA
    DEFAULT_TEMPLATE = enum.Template.EXPORTED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
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

    estimator = None  # type: KNeighborsClassifierClass

    def __init__(self, estimator: KNeighborsClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Is the estimator fitted?
        try:
            est.classes_
        except AttributeError:
            raise exception.NotFittedEstimatorError(self.estimator_name)

        if est.weights != 'uniform':
            msg = 'Only `uniform` weights are supported.'
            raise exception.NotSupportedYetError(msg)

        self.meta_info = dict(
            n_classes=len(est.classes_),
            n_templates=len(est._fit_X),  # pylint: disable=W0212
            n_features=len(est._fit_X[0]),  # pylint: disable=W0212
            metric=est.metric
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        L.opt(lazy=True).debug('Meta info: {}'.format(self.meta_info))

        self.model_data = dict(
            X=est._fit_X.tolist(),  # pylint: disable=W0212
            y=est._y.astype(int).tolist(),  # pylint: disable=W0212
            k=est.n_neighbors,  # number of relevant neighbors
            n=len(est.classes_),  # number of classes
            power=est.p
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
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_int = tpls.get_template('int').render()
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        # Make 'attached' variant:
        x_val = self.model_data.get('X')
        x_str = tpl_arr_2.render(
            type=tpl_double,
            name='X',
            values=', '.join(
                list(
                    tpl_in_brackets.render(
                        value=', '.join(list(map(converter, v)))
                    ) for v in x_val
                )
            ),
            n=len(x_val),
            m=len(x_val[0])
        )

        y_val = list(map(str, self.model_data.get('y')))
        y_str = tpl_arr_1.render(
            type=tpl_int, name='y', values=', '.join(y_val), n=len(y_val)
        )

        tpl_class = tpls.get_template('attached.class')
        plas.update(
            dict(
                X=x_str,
                y=y_str,
                k=self.model_data.get('k'),
                n=self.model_data.get('n'),
                power=self.model_data.get('power'),
            )
        )
        out_class = tpl_class.render(**plas)
        return out_class
