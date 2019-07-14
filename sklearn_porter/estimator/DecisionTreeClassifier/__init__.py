# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional, Callable
from json import encoder, dumps
from textwrap import indent
from copy import deepcopy
from logging import DEBUG

from jinja2 import Environment
from sklearn.tree.tree import DecisionTreeClassifier \
    as DecisionTreeClassifierClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template, ALL_METHODS
from sklearn_porter.exceptions import NotFittedEstimatorError
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class DecisionTreeClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a DecisionTreeClassifier classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_TEMPLATE = Template.ATTACHED
    DEFAULT_METHOD = Method.PREDICT

    SUPPORT = {
        Language.C: {
            Template.ATTACHED: ALL_METHODS,
            Template.COMBINED: ALL_METHODS,
        },
        Language.GO: {
            Template.ATTACHED: ALL_METHODS,
            Template.COMBINED: ALL_METHODS,
        },
        Language.JAVA: {
            Template.ATTACHED: ALL_METHODS,
            Template.COMBINED: ALL_METHODS,
            Template.EXPORTED: ALL_METHODS,
        },
        Language.JS: {
            Template.ATTACHED: ALL_METHODS,
            Template.COMBINED: ALL_METHODS,
        },
        Language.PHP: {
            Template.ATTACHED: ALL_METHODS,
            Template.COMBINED: ALL_METHODS,
        },
        Language.RUBY: {
            Template.ATTACHED: ALL_METHODS,
            Template.COMBINED: ALL_METHODS,
        }
    }

    estimator = None  # type: DecisionTreeClassifierClass

    def __init__(self, estimator: DecisionTreeClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # Is the estimator fitted?
        try:
            getattr(est, 'n_features_')  # for sklearn >  0.19
            getattr(est.tree_, 'value')  # for sklearn <= 0.18
        except AttributeError:
            raise NotFittedEstimatorError(self.estimator_name)

        # Extract and save meta information:
        self.meta_info = dict(
            n_features=est.n_features_,
            n_classes=len(est.tree_.value.tolist()[0][0]),
        )
        L.info('Meta info (keys): {}'.format(
            self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Meta info: {}'.format(self.meta_info))

        # Extract and save model data:
        self.model_data = dict(
            lefts=est.tree_.children_left.tolist(),
            rights=est.tree_.children_right.tolist(),
            thresholds=est.tree_.threshold.tolist(),
            indices=est.tree_.feature.tolist(),
            classes=[[int(c) for c in l[0]] for l in est.tree_.value.tolist()],
        )
        L.info('Model data (keys): {}'.format(
            self.model_data.keys()))
        if L.isEnabledFor(DEBUG):
            L.debug('Model data: {}'.format(self.model_data))

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

        # Arguments:
        kwargs.setdefault('method_name', method.value)
        converter = kwargs.get('converter')

        # Placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(dict(
            class_name=kwargs.get('class_name'),
            method_name=kwargs.get('method_name'),
        ))
        plas.update(self.meta_info)

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Export variant:
        if template == Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_int = tpls.get_template('int').render()
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        # Make contents:
        lefts_val = list(map(str, self.model_data.get('lefts')))
        lefts_str = tpl_arr_1.render(
            type=tpl_int,
            name='lefts',
            values=', '.join(lefts_val),
            n=len(lefts_val)
        )

        rights_val = list(map(str, self.model_data.get('rights')))
        rights_str = tpl_arr_1.render(
            type=tpl_int,
            name='rights',
            values=', '.join(rights_val),
            n=len(rights_val)
        )

        thresholds_val = list(map(converter, self.model_data.get('thresholds')))
        thresholds_str = tpl_arr_1.render(
            type=tpl_double,
            name='thresholds',
            values=', '.join(thresholds_val),
            n=len(thresholds_val)
        )

        indices_val = list(map(str, self.model_data.get('indices')))
        indices_str = tpl_arr_1.render(
            type=tpl_int,
            name='indices',
            values=', '.join(indices_val),
            n=len(indices_val)
        )

        classes_val = [list(map(str, e))
                       for e in self.model_data.get('classes')]
        classes_str = [', '.join(e) for e in classes_val]
        classes_str = ', '.join([tpl_in_brackets.render(value=e)
                                 for e in classes_str])
        classes_str = tpl_arr_2.render(
            type=tpl_int,
            name='classes',
            values=classes_str,
            n=len(classes_val),
            m=len(classes_val[0])
        )

        plas.update(dict(
            lefts=lefts_str,
            rights=rights_str,
            thresholds=thresholds_str,
            indices=indices_str,
            classes=classes_str,
        ))

        # Attached variant:
        if template == Template.ATTACHED:
            tpl_class = tpls.get_template('attached.class')
            out_class = tpl_class.render(**plas)
            return out_class

        # Combined variant:
        if template == Template.COMBINED:
            tpl_class = tpls.get_template('combined.class')
            out_tree = self._create_tree(tpls, language, converter)
            plas.update(dict(tree=out_tree))
            out_class = tpl_class.render(**plas)
            return out_class

    def _create_tree(
            self,
            tpls: Environment,
            language: Language,
            converter: Callable[[object], str]
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
        n_indents = 1 if language in {
            Language.JAVA,
            Language.JS,
            Language.PHP,
            Language.RUBY
        } else 0
        return self._create_branch(
            tpls, language, converter,
            self.model_data.get('lefts'),
            self.model_data.get('rights'),
            self.model_data.get('thresholds'),
            self.model_data.get('classes'),
            self.model_data.get('indices'),
            0, n_indents
        )

    def _create_branch(
            self,
            tpls: Environment,
            language: Language,
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
        if threshold[node] != -2.:
            out += '\n'
            val_a = 'features[{}]'.format(features[node])
            if language is Language.PHP:
                val_a = '$' + val_a
            val_b = converter(threshold[node])
            tpl_if = tpls.get_template('if')
            out_if = tpl_if.render(a=val_a, op='<=', b=val_b)
            out_if = indent(out_if, depth * out_indent)
            out += out_if

            if left_nodes[node] != -1.:
                out += self._create_branch(
                    tpls, language, converter, left_nodes, right_nodes,
                    threshold, value, features, left_nodes[node], depth + 1)

            out += '\n'
            out_else = tpls.get_template('else').render()
            out_else = indent(out_else, depth * out_indent)
            out += out_else

            if right_nodes[node] != -1.:
                out += self._create_branch(
                    tpls, language, converter, left_nodes, right_nodes,
                    threshold, value, features, right_nodes[node], depth + 1)

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
