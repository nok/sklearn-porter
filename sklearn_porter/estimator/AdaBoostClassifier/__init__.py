# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional, Callable
from json import encoder, dumps
from textwrap import indent
from copy import deepcopy

from sklearn.ensemble.weight_boosting import AdaBoostClassifier \
    as AdaBoostClassifierClass
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import (
    Method,
    Language,
    Template
)
from sklearn_porter.exceptions import (
    NotSupportedYetError,
    NotFittedEstimatorError
)
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class AdaBoostClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port an AdaBoostClassifier classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_METHOD = Method.PREDICT
    DEFAULT_TEMPLATE = Template.COMBINED

    SUPPORT = {
        Language.C: {Method.PREDICT: {Template.COMBINED, }},
        Language.JAVA: {Method.PREDICT: {Template.COMBINED, Template.EXPORTED}},
        Language.JS: {Method.PREDICT: {Template.COMBINED, Template.EXPORTED}},
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
        if not isinstance(est.base_estimator, DecisionTreeClassifier):
            msg = 'The used base estimator `{}` is not supported yet.'
            msg = msg.format(est.base_estimator.__class__.__qualname__)
            raise NotSupportedYetError(msg)

        # Check number of base estimators:
        if not estimator.n_estimators > 0:
            raise NotFittedEstimatorError(self.estimator_name)

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
        L.info(self.meta_info)

        self.model_data['estimators'] = []
        for e in estimators:
            self.model_data['estimators'].append(dict(
                lefts=e.tree_.children_left.tolist(),
                rights=e.tree_.children_right.tolist(),
                thresholds=e.tree_.threshold.tolist(),
                classes=[c[0] for c in e.tree_.value.tolist()],
                indices=e.tree_.feature.tolist()
            ))

    def port(
            self,
            method: Optional[Method] = None,
            language: Optional[Language] = None,
            template: Optional[Template] = None,
            **kwargs
    ) -> Union[str, Tuple[str, str]]:
        """
        Port an estimator.

        Parameters
        ----------
        method : Method
            The required method.
        language : Language
            The required language.
        template : Template
            The required template.
        kwargs

        Returns
        -------
        out_class : str
            The ported estimator.
        """
        method, language, template = self.check(
            method=method, language=language, template=template)

        # Handle optional arguments:
        kwargs.setdefault('method_name', method.value)
        converter = kwargs.get('converter')

        # Set placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(dict(
            class_name=kwargs.get('class_name'),
            method_name=kwargs.get('method_name'),
        ))
        plas.update(self.meta_info)

        # Load templates:
        tpls = self._load_templates(language.value.KEY)

        # Handle EXPORT template:
        support_export = (Language.JAVA, Language.JS)
        if method == Template.EXPORTED and language in support_export:
            tpl_class = tpls.get('exported.class')
            out_class = tpl_class.format(**plas)
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = self.model_data['estimators']
            model_data = dumps(model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_indent = tpls.get('indent')

        # Handle COMBINED template:
        out_fns = []
        for idx, model_data in enumerate(self.model_data['estimators']):
            out_fn = self._create_method(
                templates=tpls,
                language=language.value.KEY,
                converter=converter,
                method_name=plas.get('method_name') + '_' + str(idx),
                model_data=model_data)
            out_fns.append(out_fn)
        out_fns = '\n'.join(out_fns)

        # Generate function names:
        out_fn_calls = ''
        if language in (Language.JAVA,):
            out_fn_calls = []

            tpl_call = tpls.get('combined.method_calls')
            for idx in range(self.meta_info.get('n_estimators')):
                plas_copy = deepcopy(plas)
                plas_copy.update(dict(method_index=idx))
                out_call = tpl_call.format(**plas_copy)
                out_fn_calls.append(out_call)
            out_fn_calls = '\n'.join(out_fn_calls)

            n_indents = 1
            out_fn_calls = indent(out_fn_calls, n_indents * tpl_indent)
            out_fn_calls = out_fn_calls[(n_indents * len(tpl_indent)):]

        # Make method:
        tpl_method = tpls.get('combined.method')
        plas_copy = deepcopy(plas)
        plas_copy.update(dict(
            methods=out_fns,
            method_calls=out_fn_calls
        ))
        out_method = tpl_method.format(**plas_copy)

        if language in (Language.JAVA, Language.JS):
            n_indents = 1
            out_method = indent(out_method, n_indents * tpl_indent)

        # Make class:
        tpl_class = tpls.get('combined.class')
        copy_plas = deepcopy(plas)
        copy_plas.update(dict(method=out_method))
        out_class = tpl_class.format(**copy_plas)

        return out_class

    def _create_method(
            self,
            templates: dict,
            language: str,
            converter: Callable[[object], str],
            method_name: str,
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
            templates,
            language,
            converter,
            model_data.get('lefts'),
            model_data.get('rights'),
            model_data.get('thresholds'),
            model_data.get('classes'),
            model_data.get('indices'),
            0, 1
        )

        tpl_tree = templates.get('combined.tree')
        out_tree = tpl_tree.format(
            method_name=method_name,
            methods=tree_branches,
            n_classes=self.meta_info.get('n_classes')
        )
        return out_tree

    def _create_branch(
            self,
            tpls: dict,
            language: str,
            converter: Callable[[object], str],
            left_nodes: list,
            right_nodes: list,
            threshold: list,
            value: list,
            features: list,
            node: int,
            depth: int
    ):
        """
        The ported single tree as function or method.

        Parameters
        ----------
        tpls : Dict[str, str]
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
        temp_indent = tpls.get('indent')
        if threshold[node] != -2.:

            out += '\n'
            tpl = tpls.get('if')
            tpl = indent(tpl, depth * temp_indent)
            val_1 = 'features[{}]'.format(features[node])
            if language == 'php':
                val_1 = '$' + val_1
            val_2 = converter(threshold[node])
            out += tpl.format(val_1, '<=', val_2)

            if left_nodes[node] != -1.:
                out += self._create_branch(
                    tpls, language, converter, left_nodes, right_nodes,
                    threshold, value, features, left_nodes[node], depth + 1)

            out += '\n'
            tpl = tpls.get('else')
            tpl = indent(tpl, depth * temp_indent)
            out += tpl

            if right_nodes[node] != -1.:
                out += self._create_branch(
                    tpls, language, converter, left_nodes, right_nodes,
                    threshold, value, features, right_nodes[node], depth + 1)

            out += '\n'
            tpl = tpls.get('endif')
            tpl = indent(tpl, depth * temp_indent)
            out += tpl
        else:
            clazzes = []
            tpl = 'classes[{0}] = {1}'
            if language == 'php':
                tpl = '$' + tpl
            tpl = indent(tpl, depth * temp_indent)

            for i, rate in enumerate(value[node]):
                clazz = tpl.format(i, rate)
                clazz = '\n' + clazz
                clazzes.append(clazz)

            tpl = tpls.get('join')
            out += tpl.join(clazzes) + tpl
        return out
