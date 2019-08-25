# -*- coding: utf-8 -*-

from copy import deepcopy
from json import dumps, encoder
from textwrap import indent
from typing import Callable, Optional, Tuple, Union

from jinja2 import Environment
from loguru import logger as L

# scikit-learn
from sklearn.ensemble.weight_boosting import \
    AdaBoostClassifier as AdaBoostClassifierClass
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter.enums import Language, Method, Template, ALL_METHODS
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.exceptions import NotSupportedYetError


class AdaBoostClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port an AdaBoostClassifier classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_TEMPLATE = Template.COMBINED
    DEFAULT_METHOD = Method.PREDICT

    SUPPORT = {
        Language.C: {
            Template.COMBINED: {
                Method.PREDICT,
            },
        },
        Language.JAVA: {
            Template.COMBINED: {
                Method.PREDICT,
            },
            Template.EXPORTED: {
                Method.PREDICT,
            },
        },
        Language.JS: {
            Template.COMBINED: {
                Method.PREDICT,
            },
            Template.EXPORTED: ALL_METHODS,
        },
    }

    estimator = None  # type: AdaBoostClassifierClass

    def __init__(self, estimator: AdaBoostClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Check the used algorithm type:
        if est.algorithm != 'SAMME.R':
            msg = 'The used algorithm `{}` is not supported yet.'
            msg = msg.format(est.algorithm)
            raise NotSupportedYetError(msg)

        # Check type of base estimators:
        if not isinstance(est.base_estimator_, DecisionTreeClassifier):
            msg = 'The used base estimator `{}` is not supported yet.'
            msg = msg.format(est.base_estimator.__class__.__qualname__)
            raise NotSupportedYetError(msg)

        # Ignore estimators with zero weight:
        estimators = []
        for idx in range(est.n_estimators):
            if est.estimator_weights_[idx] > 0:
                estimators.append(est.estimators_[idx])
        n_estimators = len(estimators)

        self.meta_info = dict(
            n_classes=est.n_classes_,
            n_features=est.estimators_[0].n_features_,
            n_estimators=n_estimators,
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        L.opt(lazy=True).debug('Meta info: {}'.format(self.meta_info))

        self.model_data['estimators'] = []
        for e in estimators:
            self.model_data['estimators'].append(
                dict(
                    lefts=e.tree_.children_left.tolist(),
                    rights=e.tree_.children_right.tolist(),
                    thresholds=e.tree_.threshold.tolist(),
                    classes=[c[0] for c in e.tree_.value.tolist()],
                    indices=e.tree_.feature.tolist()
                )
            )
        L.info('Model data (keys): {}'.format(self.model_data.keys()))
        L.opt(lazy=True).debug('Model data: {}'.format(self.model_data))

    def port(
        self,
        language: Optional[Language] = None,
        template: Optional[Template] = None,
        to_json: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, str]]:
        """
        Port an estimator.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        to_json : bool (default: False)
            Return the result as JSON string.
        kwargs

        Returns
        -------
        out_class : str
            The ported estimator.
        """
        method, language, template = self.check(
            language=language, template=template
        )

        # Default arguments:
        kwargs.setdefault('method_name', method.value)
        converter = kwargs.get('converter')

        # Placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(
            dict(
                class_name=kwargs.get('class_name'),
                method_name=kwargs.get('method_name'),
                to_json=to_json,
            )
        )
        plas.update(self.meta_info)

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Export:
        if template == Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = self.model_data.get('estimators')
            model_data = dumps(model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_indent = tpls.get_template('indent')

        # Handle COMBINED template:
        out_fns = []
        for idx, model_data in enumerate(self.model_data.get('estimators')):
            out_fn = self._create_method(
                templates=tpls,
                language=language.value.KEY,
                converter=converter,
                method_name=plas.get('method_name') + '_' + str(idx),
                model_data=model_data
            )
            out_fns.append(out_fn)
        out_fns = '\n'.join(out_fns)

        # Generate function names:
        tpl_calls = tpls.get_template('combined.method_calls')
        out_calls = []
        for idx in range(self.meta_info.get('n_estimators')):
            plas_copy = deepcopy(plas)
            plas_copy.update(dict(method_index=idx))
            out_call = tpl_calls.format(**plas_copy)
            out_calls.append(out_call)
        out_calls = '\n'.join(out_calls)

        n_indents = 1
        out_calls = indent(out_calls, n_indents * tpl_indent)
        out_calls = out_calls[(n_indents * len(tpl_indent)):]

        # Make method:
        tpl_method = tpls.get_template('combined.method')
        plas_copy = deepcopy(plas)
        plas_copy.update(dict(methods=out_fns, method_calls=out_calls))
        out_method = tpl_method.format(**plas_copy)

        if language in (Language.JAVA, Language.JS):
            n_indents = 1
            out_method = indent(out_method, n_indents * tpl_indent)

        # Make class:
        tpl_class = tpls.get_template('combined.class')
        copy_plas = deepcopy(plas)
        copy_plas.update(dict(method=out_method))
        out_class = tpl_class.format(**copy_plas)

        return out_class

    def _create_method(
        self,
        templates: Environment,
        language: str,
        converter: Callable[[object], str],
        method_name: str,
        model_data: dict,
    ):
        """
        Port a method for a single tree.

        Parameters
        ----------
        templates : Environment
            All relevant templates.
        language
            The required language.
        converter
            The number converter.
        method_name : int
            The name of the single decision tree.
        model_data : dict
            The model data of the single decision tree.

        Returns
        -------
        out_tree : str
            The created method as string.
        """
        tree_branches = self._create_branch(
            templates, language, converter, model_data.get('lefts'),
            model_data.get('rights'), model_data.get('thresholds'),
            model_data.get('classes'), model_data.get('indices'), 0, 1
        )

        tpl_tree = templates.get_template('combined.single_method')
        out_tree = tpl_tree.format(
            method_name=method_name,
            methods=tree_branches,
            n_classes=self.meta_info.get('n_classes')
        )
        return out_tree

    def _create_tree(
        self,
        tpls: Environment,
        language: Language,
        converter: Callable[[object], str],
    ):
        """
        Build a decision tree.

        Parameters
        ----------
        tpls : Environment
            All relevant templates.
        language : str
            The required language.
        converter : Callable[[object], str]
            The number converter.

        Returns
        -------
        A tree of a DecisionTreeClassifier.
        """
        n_indents = (
            1 if language in {
                Language.JAVA, Language.JS, Language.PHP, Language.RUBY
            } else 0
        )
        return self._create_branch(
            tpls,
            language,
            converter,
            self.model_data.get('lefts'),
            self.model_data.get('rights'),
            self.model_data.get('thresholds'),
            self.model_data.get('classes'),
            self.model_data.get('indices'),
            0,
            n_indents,
        )

    def _create_method(
        self,
        templates: Environment,
        language: str,
        converter: Callable[[object], str],
        method_index: str,
        class_name: str,
        model_data: dict,
    ):
        """
        Port a method for a single tree.

        Parameters
        ----------
        templates : Dict[str, str]
            All relevant templates.
        language
            The required language.
        converter
            The number converter.
        method_index : str
            The index of a single decision tree.
        class_name : str
            The class name.
        model_data : dict
            The model data of the single decision tree.

        Returns
        -------
        out_tree : str
            The created method as string.
        """
        tree = self._create_tree(
            templates, language, converter, model_data.get('lefts'),
            model_data.get('rights'), model_data.get('thresholds'),
            model_data.get('classes'), model_data.get('indices'), 0, 1
        )

        tpl_tree = templates.get_template('combined.tree')
        out_tree = tpl_tree.render(
            tree=tree,
            n_classes=self.meta_info.get('n_classes'),
            n_estimators=self.meta_info.get('n_estimators'),
            method_index=method_index,
            class_name=class_name
        )
        return out_tree

    def _create_tree(
        self, tpls: Environment, language: str,
        converter: Callable[[object], str], left_nodes: list, right_nodes: list,
        threshold: list, value: list, features: list, node: int, depth: int
    ):
        """
        The ported single tree as function or method.

        Parameters
        ----------
        tpls : Environment
            All relevant templates.
        language
            The required language.
        converter
            The number converter.
        left_nodes : list
            The left children node.
        right_nodes : list
            The left children node.
        threshold : list
            The decision thresholds.
        value : list
            The label or class.
        features : list
            The feature values.
        node : list
            The current node.
        depth : list
            The tree depth.

        Returns
        -------
        A single branch of a DecisionTreeClassifier.
        """
        out = ''
        out_indent = tpls.get_template('indent').render()
        if threshold[node] != -2.0:
            if node != 0:
                out += '\n'

            val_a = 'features[{}]'.format(features[node])
            if language is Language.PHP:
                val_a = '$' + val_a
            val_b = converter(threshold[node])
            tpl_if = tpls.get_template('if')
            out_if = tpl_if.render(a=val_a, op='<=', b=val_b)
            if node != 0:
                out_if = indent(out_if, depth * out_indent)
            out += out_if

            if left_nodes[node] != -1.0:
                out += self._create_tree(
                    tpls,
                    language,
                    converter,
                    left_nodes,
                    right_nodes,
                    threshold,
                    value,
                    features,
                    left_nodes[node],
                    depth + 1,
                )

            out += '\n'
            out_else = tpls.get_template('else').render()
            out_else = indent(out_else, depth * out_indent)
            out += out_else

            if right_nodes[node] != -1.0:
                out += self._create_tree(
                    tpls,
                    language,
                    converter,
                    left_nodes,
                    right_nodes,
                    threshold,
                    value,
                    features,
                    right_nodes[node],
                    depth + 1,
                )

            out += '\n'
            out_endif = tpls.get_template('endif').render()
            out_endif = indent(out_endif, depth * out_indent)
            out += out_endif
        else:
            clazzes = []
            tpl = 'classes[{0}] = {1}'
            if language is Language.PHP:
                tpl = '$' + tpl
            tpl = indent(tpl, depth * out_indent)

            for i, rate in enumerate(value[node]):
                if int(rate) > 0:
                    clazz = tpl.format(i, rate)
                    clazz = '\n' + clazz
                    clazzes.append(clazz)

            out_join = tpls.get_template('join').render()
            out += out_join.join(clazzes) + out_join
        return out
