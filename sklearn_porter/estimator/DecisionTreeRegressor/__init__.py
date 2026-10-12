from collections.abc import Callable
from copy import deepcopy
from json import dumps
from textwrap import indent

from jinja2 import Environment
from loguru import logger as L

# scikit-learn
from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegressorClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.DecisionTreeClassifier import (
    DecisionTreeClassifier,
)
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.utils import get_thresholds


class DecisionTreeRegressor(DecisionTreeClassifier, EstimatorBase):
    """Extract model data and port a DecisionTreeRegressor regressor."""

    SKLEARN_URL = 'sklearn.tree.DecisionTreeRegressor.html'

    DEFAULT_LANGUAGE = enum.Language.JS
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.GO: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
    }

    # The key of the model data with the content of the leaves:
    LEAF_DATA = 'values'

    estimator = None  # type: DecisionTreeRegressorClass

    def __init__(self, estimator: DecisionTreeRegressorClass):
        EstimatorBase.__init__(self, estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Is the estimator fitted?
        try:
            est.tree_
        except AttributeError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        # Extract and save meta information:
        self.meta_info = dict(
            n_features=est.n_features_in_,
            n_outputs=est.n_outputs_,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        # Extract and save model data:
        # The prediction of a leaf is stored in `tree_.value` of shape
        # (n_nodes, n_outputs, 1), e.g. the (weighted) mean of the targets
        # (`squared_error`, `friedman_mse`, `poisson`) or the (weighted)
        # median of the targets (`absolute_error`).
        self.model_data = dict(
            lefts=est.tree_.children_left.tolist(),
            rights=est.tree_.children_right.tolist(),
            thresholds=get_thresholds(est.tree_),
            indices=est.tree_.feature.tolist(),
            values=est.tree_.value[:, :, 0].tolist(),
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
            used, e.g. `DecisionTreeRegressor`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        out_class : str
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

        # Make 'combined' variant:
        if template == enum.Template.COMBINED:
            tpl_class = tpls.get_template('combined.class')
            out_tree = self._create_tree(tpls, language, converter)
            plas.update(dict(tree=out_tree))
            out_class = tpl_class.render(**plas)
            return out_class

        # Make 'attached' variant:
        tpl_int = tpls.get_template('int').render()
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        for name in ('lefts', 'rights', 'indices'):
            vals = list(map(str, self.model_data.get(name)))
            plas[name] = tpl_arr_1.render(
                type=tpl_int,
                name=name,
                values=', '.join(vals),
                n=len(vals),
            )

        thresholds_val = list(map(converter, self.model_data.get('thresholds')))
        plas['thresholds'] = tpl_arr_1.render(
            type=tpl_double,
            name='thresholds',
            values=', '.join(thresholds_val),
            n=len(thresholds_val),
        )

        values_val = [
            list(map(converter, e)) for e in self.model_data.get('values')
        ]
        values_str = ', '.join(
            [tpl_in_brackets.render(value=', '.join(e)) for e in values_val]
        )
        plas['values'] = tpl_arr_2.render(
            type=tpl_double,
            name='values',
            values=values_str,
            n=len(values_val),
            m=len(values_val[0]),
        )

        tpl_class = tpls.get_template('attached.class')
        out_class = tpl_class.render(**plas)
        return out_class

    def _create_leaf(
        self,
        tpls: Environment,
        language: enum.Language,
        converter: Callable[[object], str],
        value: list,
        depth: int,
    ):
        """
        The assignments of a single leaf.

        Parameters
        ----------
        tpls : Environment
            All relevant templates.
        language
            The required language.
        converter
            The number converter.
        value : list
            The predicted values of the leaf.
        depth : list
            The tree depth.

        Returns
        -------
        The assignments of the predicted values of a single leaf.
        """
        out_indent = tpls.get_template('indent').render()
        tpl = 'values[{0}] = {1}'
        if language is enum.Language.PHP:
            tpl = '$' + tpl
        tpl = indent(tpl, depth * out_indent)
        out_join = tpls.get_template('join').render()
        assignments = [
            '\n' + tpl.format(i, converter(v)) for i, v in enumerate(value)
        ]
        return out_join.join(assignments) + out_join
