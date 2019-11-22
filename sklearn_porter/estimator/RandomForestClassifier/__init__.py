# -*- coding: utf-8 -*-

from copy import deepcopy
from json import dumps, encoder
from textwrap import indent
from typing import Callable, Dict, Optional, Tuple, Union

from loguru import logger as L

# scikit-learn
from jinja2 import Environment
from sklearn.ensemble.forest import \
    RandomForestClassifier as RandomForestClassifierClass
from sklearn.tree import DecisionTreeClassifier

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class RandomForestClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a RandomForestClassifier classifier."""

    SKLEARN_URL = 'sklearn.ensemble.RandomForestClassifier.html'

    DEFAULT_LANGUAGE = enum.Language.JAVA
    DEFAULT_TEMPLATE = enum.Template.COMBINED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        # Language.C: {
        #     Template.COMBINED: {},
        # },
        # Language.GO: {
        #     Template.COMBINED: {},
        # },
        enum.Language.JAVA: {
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
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        # Language.RUBY: {
        #     Template.COMBINED: {},
        # },
    }

    estimator = None  # type: RandomForestClassifierClass

    def __init__(self, estimator: RandomForestClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Check type of base estimators:
        if not isinstance(est.base_estimator, DecisionTreeClassifier):
            msg = 'The used base estimator `{}` is not supported yet.'
            msg = msg.format(est.base_estimator.__class__.__qualname__)
            raise exception.NotSupportedYetError(msg)

        # Check number of base estimators:
        if not estimator.n_estimators > 0:
            raise exception.NotFittedEstimatorError(self.estimator_name)

        self.estimators = [
            est.estimators_[idx] for idx in range(est.n_estimators)
        ]
        self.n_estimators = len(self.estimators)
        self.n_features = est.estimators_[0].n_features_
        self.n_classes = est.n_classes_

        # Extract and save meta information:
        self.meta_info = dict(
            n_estimators=est.n_estimators,
            n_classes=est.n_classes_,
            n_features=est.estimators_[0].n_features_,
        )
        L.info('Meta info (keys): {}'.format(self.meta_info.keys()))
        L.opt(lazy=True).debug('Meta info: {}'.format(self.meta_info))

        # Extract and save model data:
        self.model_data['estimators'] = []
        for e in est.estimators_:
            self.model_data['estimators'].append(
                dict(
                    lefts=e.tree_.children_left.tolist(),
                    rights=e.tree_.children_right.tolist(),
                    thresholds=e.tree_.threshold.tolist(),
                    classes=[c[0] for c in e.tree_.value.astype(int).tolist()],
                    indices=e.tree_.feature.tolist()
                )
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
        encoder.FLOAT_REPR = lambda o: converter(o)

        # Make 'exported' variant:
        if template == enum.Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            model_data = self.model_data.get('estimators')
            model_data = dumps(model_data, separators=(',', ':'))
            return out_class, model_data

        # Make 'attached' variant:
        if template == enum.Template.ATTACHED:
            tpl_class = tpls.get_template('attached.class')
            tpl_init = tpls.get_template('init')
            model_data = self.model_data.get('estimators')
            model_data = dumps(model_data, separators=(',', ':'))

            if language is enum.Language.PHP:
                model_data = "json_decode('" + model_data + "', true)"

            plas['model'] = tpl_init.render(name='model', value=model_data)
            out_class = tpl_class.render(**plas)
            return out_class

        # Make 'combined' variant:
        # Pick templates:
        tpl_indent = tpls.get_template('indent').render()

        # Generate functions:
        out_fns = []
        for idx, model_data in enumerate(self.model_data.get('estimators')):
            out_fn = self._create_method(
                templates=tpls,
                language=language.value.KEY,
                converter=converter,
                method_index=str(idx),
                class_name=plas.get('class_name'),
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
            out_call = tpl_calls.render(**plas_copy)
            out_calls.append(out_call)
        out_calls = '\n'.join(out_calls)

        n_indents = 1
        out_calls = indent(out_calls, n_indents * tpl_indent)
        out_calls = out_calls[(n_indents * len(tpl_indent)):]

        # Make method:
        tpl_method = tpls.get_template('combined.method')
        plas_copy = deepcopy(plas)
        plas_copy.update(dict(methods=out_fns, method_calls=out_calls))
        out_method = tpl_method.render(**plas_copy)

        if language in (enum.Language.JAVA, enum.Language.JS):
            n_indents = 1
            out_method = indent(out_method, n_indents * tpl_indent)
            out_method = out_method[(n_indents * len(tpl_indent)):]

        # Make class:
        tpl_class = tpls.get_template('combined.class')
        copy_plas = deepcopy(plas)
        copy_plas.update(dict(method=out_method))
        out_class = tpl_class.render(**copy_plas)
        return out_class

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
            if language is enum.Language.PHP:
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
            if language is enum.Language.PHP:
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
