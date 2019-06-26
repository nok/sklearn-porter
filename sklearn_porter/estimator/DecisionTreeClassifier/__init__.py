# -*- coding: utf-8 -*-

from typing import Union, Dict, Tuple, Optional, Callable
from json import encoder, dumps
from textwrap import indent
from copy import deepcopy
from logging import DEBUG

from sklearn.tree.tree import DecisionTreeClassifier \
    as DecisionTreeClassifierClass

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class DecisionTreeClassifier(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a DecisionTreeClassifier classifier."""

    DEFAULT_LANGUAGE = Language.JAVA
    DEFAULT_METHOD = Method.PREDICT
    DEFAULT_TEMPLATE = Template.ATTACHED

    _full_support = {
        Method.PREDICT: {
            Template.COMBINED,
            Template.ATTACHED,
            Template.EXPORTED
        }
    }
    SUPPORT = {
        Language.C: _full_support,
        Language.GO: _full_support,
        Language.JAVA: _full_support,
        Language.JS: _full_support,
        Language.PHP: _full_support,
        Language.RUBY: _full_support
    }

    estimator = None  # type: DecisionTreeClassifierClass

    def __init__(self, estimator: DecisionTreeClassifierClass):
        super().__init__(estimator)
        L.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

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

        if template == Template.EXPORTED:
            tpl_class = tpls.get('exported.class')
            out_class = tpl_class.format(**plas)
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Pick templates:
        tpl_int = tpls.get('int')
        tpl_double = tpls.get('double')
        tpl_arr_1 = tpls.get('arr[]')
        tpl_arr_2 = tpls.get('arr[][]')
        tpl_in_brackets = tpls.get('in_brackets')

        # Make contents:
        lefts_val = list(map(str, self.model_data['lefts']))
        lefts_str = tpl_arr_1.format(
            type=tpl_int,
            name='lefts',
            values=', '.join(lefts_val),
            n=len(lefts_val)
        )

        rights_val = list(map(str, self.model_data['rights']))
        rights_str = tpl_arr_1.format(
            type=tpl_int,
            name='rights',
            values=', '.join(rights_val),
            n=len(rights_val)
        )

        thresholds_val = list(map(converter, self.model_data['thresholds']))
        thresholds_str = tpl_arr_1.format(
            type=tpl_double,
            name='thresholds',
            values=', '.join(thresholds_val),
            n=len(thresholds_val)
        )

        indices_val = list(map(str, self.model_data['indices']))
        indices_str = tpl_arr_1.format(
            type=tpl_int,
            name='indices',
            values=', '.join(indices_val),
            n=len(indices_val)
        )

        classes_val = [list(map(str, e)) for e in self.model_data['classes']]
        classes_str = [', '.join(e) for e in classes_val]
        classes_str = ', '.join([tpl_in_brackets.format(e)
                                 for e in classes_str])
        classes_str = tpl_arr_2.format(
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

        if template == Template.ATTACHED:
            tpl_class = tpls.get('attached.class')
            out_class = tpl_class.format(**plas)
            return out_class

        if template == Template.COMBINED:

            # Pick templates:
            tpl_indent = tpls.get('indent')
            tpl_method = tpls.get('combined.method')
            tpl_class = tpls.get('combined.class')

            # Make tree:
            out_tree = self._create_tree(tpls, language.value.KEY, converter)
            out_tree = indent(out_tree, 1 * tpl_indent)

            # Make method:
            plas.update(dict(tree=out_tree))
            n_indents = 1 if language in [
                Language.JAVA,
                Language.JS,
                Language.PHP,
                Language.RUBY
            ] else 0
            tpl_method = indent(tpl_method, n_indents * tpl_indent)
            tpl_method = tpl_method[(n_indents * len(tpl_indent)):]
            out_method = tpl_method.format(**plas)

            # Make class:
            plas.update(dict(method=out_method))
            out_class = tpl_class.format(**plas)

            return out_class

    def _create_tree(
            self,
            tpls: Dict[str, str],
            language: str,
            converter: Callable[[object], str]
    ):
        """
        Build a decision tree.

        Parameters
        ----------
        tpls : Dict[str, str]
            All relevant templates.
        language : str
            The required language.
        converter : Callable[[object], str]
            The number converter.

        Returns
        -------
        A tree of a DecisionTreeClassifier.
        """
        n_indents = 1 if language in {'java', 'js', 'php', 'ruby'} else 0
        return self._create_branch(
            tpls, language, converter,
            self.model_data['lefts'],
            self.model_data['rights'],
            self.model_data['thresholds'],
            self.model_data['classes'],
            self.model_data['indices'],
            0, n_indents
        )

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
                if int(rate) > 0:
                    clazz = tpl.format(i, rate)
                    clazz = '\n' + clazz
                    clazzes.append(clazz)

            tpl = tpls.get('join')
            out += tpl.join(clazzes) + tpl
        return out
