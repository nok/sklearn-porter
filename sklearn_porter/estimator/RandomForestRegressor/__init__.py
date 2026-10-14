from collections.abc import Callable
from copy import deepcopy
from json import dumps

from jinja2 import Environment
from loguru import logger as L

# scikit-learn
from sklearn.ensemble import (
    ExtraTreesRegressor as ExtraTreesRegressorClass,
)
from sklearn.ensemble import (
    RandomForestRegressor as RandomForestRegressorClass,
)
from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegressorClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.DecisionTreeClassifier import (
    DecisionTreeClassifier,
)
from sklearn_porter.estimator.DecisionTreeRegressor import (
    DecisionTreeRegressor,
)
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.utils import get_thresholds, has_missing_go_to_left


class RandomForestRegressor(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a RandomForestRegressor regressor."""

    SKLEARN_URL = 'sklearn.ensemble.RandomForestRegressor.html'

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

    # The single trees are generated like the ones of a DecisionTreeRegressor:
    _create_branch = DecisionTreeClassifier._create_branch
    _create_leaf = DecisionTreeRegressor._create_leaf

    def __init__(
        self,
        estimator: RandomForestRegressorClass | ExtraTreesRegressorClass,
    ):
        super().__init__(estimator)
        L.info(f'Create specific estimator `{self.estimator_name}`.')
        est = estimator  # alias

        # Is the estimator fitted?
        try:
            est.estimators_
        except AttributeError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        # Check type of base estimators:
        if not isinstance(est.estimator_, DecisionTreeRegressorClass):
            name = est.estimator_.__class__.__qualname__
            msg = f'The used base estimator `{name}` is not supported yet.'
            raise exception.NotSupportedYetError(msg)

        # Extract and save meta information:
        self.meta_info = dict(
            n_estimators=len(est.estimators_),
            n_features=est.n_features_in_,  # ty: ignore[unresolved-attribute]
            n_outputs=est.n_outputs_,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        # Extract and save model data:
        # scikit-learn sums up the predictions of all trees in their order
        # and divides the sum by the number of trees. The ported estimators
        # do the same to get exactly the same results.
        self.model_data = dict(
            estimators=[
                dict(
                    lefts=e.tree_.children_left.tolist(),
                    rights=e.tree_.children_right.tolist(),
                    thresholds=get_thresholds(e.tree_),
                    indices=e.tree_.feature.tolist(),
                    # Samples with missing values (NaN) go to the left child
                    # if 1, otherwise to the right child:
                    missing_go_to_left=e.tree_.missing_go_to_left.tolist(),
                    values=e.tree_.value[:, :, 0].tolist(),
                )
                for e in est.estimators_
            ]
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
            used, e.g. `RandomForestRegressor`.
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
        plas.update(
            has_missing_go_to_left=any(
                map(has_missing_go_to_left, self.model_data['estimators'])
            )
        )

        # Templates:
        tpls = self._load_templates(language.value.KEY)
        trees = self.model_data['estimators']

        # Make 'exported' variant:
        if template == enum.Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            model_data = dumps(trees, separators=(',', ':'))
            return out_class, model_data

        # Make 'attached' variant:
        if template == enum.Template.ATTACHED:
            tpl_class = tpls.get_template('attached.class')
            plas['model'] = self._create_model(tpls, language, converter)
            out_class = tpl_class.render(**plas)
            return out_class

        # Make 'combined' variant:
        tpl_tree = tpls.get_template('combined.tree')
        out_trees = []
        for idx, tree in enumerate(trees):
            out_tree = self._create_branch(
                tpls,
                language,
                converter,
                tree['lefts'],
                tree['rights'],
                tree['thresholds'],
                tree['values'],
                tree['indices'],
                tree['missing_go_to_left'],
                0,
                1,
            )
            out_trees.append(tpl_tree.render(index=idx, tree=out_tree, **plas))
        plas['trees'] = '\n\n'.join(out_trees)
        tpl_class = tpls.get_template('combined.class')
        out_class = tpl_class.render(**plas)
        return out_class

    def _create_model(
        self,
        tpls: Environment,
        language: enum.Language,
        converter: Callable[[object], str],
    ) -> str:
        """
        Embed the model data of all trees into the source code.

        C and Java get array literals per tree, all other languages parse
        the model data as JSON string.

        Parameters
        ----------
        tpls : Environment
            All relevant templates.
        language : Language
            The required language.
        converter : Callable[[object], str]
            The number converter.

        Returns
        -------
        The model data as source code.
        """
        trees = self.model_data['estimators']
        if language in (enum.Language.C, enum.Language.JAVA):
            tpl_tree = tpls.get_template('attached.tree')
            out_trees = []
            for idx, tree in enumerate(trees):
                out_trees.append(
                    tpl_tree.render(
                        index=idx,
                        lefts=', '.join(map(str, tree['lefts'])),
                        rights=', '.join(map(str, tree['rights'])),
                        thresholds=', '.join(
                            map(converter, tree['thresholds'])
                        ),
                        indices=', '.join(map(str, tree['indices'])),
                        missing_go_to_left=', '.join(
                            map(str, tree['missing_go_to_left'])
                        ),
                        values=[
                            ', '.join(map(converter, v)) for v in tree['values']
                        ],
                    )
                )
            return '\n'.join(out_trees)

        tpl_init = tpls.get_template('init')
        value = dumps(trees, separators=(',', ':'))
        if language is enum.Language.PHP:
            value = f"json_decode('{value}', true)"
        elif language is enum.Language.GO:
            value = f'`{value}`'
        elif language is enum.Language.RUBY:
            value = f"JSON.parse('{value}')"
        return tpl_init.render(name='model', value=value)
