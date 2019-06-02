# -*- coding: utf-8 -*-

import os
from typing import Callable, Optional, Union, List
from textwrap import indent
from pathlib import Path
from json import dumps, encoder

from sklearn.tree.tree import DecisionTreeClassifier \
    as DecisionTreeClassifierClass

from sklearn_porter.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.utils import get_logger

from logging import DEBUG
L = get_logger(__name__)


class DecisionTreeClassifier(EstimatorBase, EstimatorApiABC):
    """
    Extract model data and port a DecisionTreeClassifier classifier.

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    supported_languages = {'java'}
    supported_methods = {'predict'}
    supported_templates = {'combined', 'attached', 'exported'}

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
        L.info('Extracted meta information (keys only): {}'.format(
            self.meta_info.keys()))
        if L.isEnabledFor(DEBUG):
            dump = EstimatorBase._dump_dict(self.meta_info)
            L.debug('Extracted meta information:\n{}'.format(dump))

        # Extract and save model data:
        self.model_data = dict(
            lefts=est.tree_.children_left.tolist(),
            rights=est.tree_.children_right.tolist(),
            thresholds=est.tree_.threshold.tolist(),
            indices=est.tree_.feature.tolist(),
            classes=[[int(c) for c in l[0]] for l in est.tree_.value.tolist()],
        )
        L.info('Extracted model data (keys only): {}'.format(
            self.model_data.keys()))
        if L.isEnabledFor(DEBUG):
            dump = EstimatorBase._dump_dict(self.model_data)
            L.debug('Extracted model data:\n{}'.format(dump))

    def port(
            self,
            method: str = 'predict',
            language: str = 'java',
            template: str = 'combined',
            **kwargs
    ) -> str:
        super().check_arguments(method, language, template)

        converter = kwargs.get('converter')

        # Placeholders:
        placeholders = dict(
            class_name=kwargs.get('class_name'),
            method_name=kwargs.get('method_name'),
        )
        placeholders.update({  # merge all placeholders
            **self.model_data,
            **self.meta_info
        })

        # Load templates:
        temps = self._load_templates(language)

        if template == 'exported':
            return temps.get('exported.class').format(**placeholders)

        # Pick templates:
        temp_int = temps.get('int')
        temp_double = temps.get('double')
        temp_arr_1 = temps.get('arr[]')
        temp_arr_2 = temps.get('arr[][]')
        temp_in_brackets = temps.get('in_brackets')

        # Make contents:
        lefts = list(map(str, self.model_data['lefts']))
        lefts = temp_arr_1.format(type=temp_int, name='lefts',
                                  values=', '.join(lefts),
                                  n=len(lefts))

        rights = list(map(str, self.model_data['rights']))
        rights = temp_arr_1.format(type=temp_int, name='rights',
                                   values=', '.join(rights),
                                   n=len(rights))

        thresholds = list(map(converter, self.model_data['thresholds']))
        thresholds = temp_arr_1.format(type=temp_double, name='thresholds',
                                       values=', '.join(thresholds),
                                       n=len(thresholds))

        indices = list(map(str, self.model_data['indices']))
        indices = temp_arr_1.format(type=temp_int, name='indices',
                                    values=', '.join(indices), n=len(indices))

        classes = [list(map(str, e)) for e in self.model_data['classes']]
        n, m = len(classes), self.meta_info.get('n_classes')
        classes = [', '.join(e) for e in classes]
        classes = ', '.join([temp_in_brackets.format(e) for e in classes])
        classes = temp_arr_2.format(type=temp_int, name='classes',
                                    values=classes, n=n, m=m)

        placeholders.update(dict(
            lefts=lefts,
            rights=rights,
            thresholds=thresholds,
            indices=indices,
            classes=classes,
        ))

        if template == 'attached':
            return temps.get('attached.class').format(**placeholders)

        if template == 'combined':

            # Pick templates:
            temp_indent = temps.get('indent')
            temp_method = temps.get('combined.method')
            temp_class = temps.get('combined.class')

            # Make tree:
            made_tree = self._create_tree(temps, language, converter)
            made_tree = indent(made_tree, 1 * temp_indent)

            # Make method:
            n_indents = 1 if language in {'java', 'js', 'php', 'ruby'} else 0
            temp_method = indent(temp_method, n_indents * temp_indent)
            temp_method = temp_method[(n_indents * len(temp_indent)):]
            placeholders.update(dict(tree=made_tree))
            made_method = temp_method.format(**placeholders)

            # Make class:
            placeholders.update(dict(method=made_method))
            made_class = temp_class.format(**placeholders)

            return made_class

    def dump(
            self,
            method: str = 'predict',
            language: str = 'java',
            template: str = 'combined',
            directory: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Union[str, List[str]]:

        if not directory:
            directory = Path(os.getcwd()).resolve()
        if isinstance(directory, str):
            directory = Path(directory)
        if not directory.is_dir():
            directory = directory.parent

        ported = self.port(method, language, template, **kwargs)

        class_name = kwargs.get('class_name')

        package = 'sklearn_porter.language.' + language
        name = 'SUFFIX'
        suffix = getattr(__import__(package, fromlist=[name]), name)

        filename = class_name + '.' + suffix

        filepath = directory / filename
        filepath.write_text(ported, encoding='utf-8')

        paths = str(filepath)

        if template == 'exported':
            converter = kwargs.get('converter')
            encoder.FLOAT_REPR = lambda o: converter(o)
            json_data = dumps(self.model_data, sort_keys=True)
            json_path = directory / (class_name + '.json')
            json_path.write_text(json_data, encoding='utf-8')
            paths = [paths, str(json_path)]

        return paths

    def _create_tree(
            self,
            templates: dict,
            language: str,
            converter: Callable[[object], str]
    ):
        """
        Parse and build the tree branches.

        Returns
        -------
        :return : string
            The tree branches as string.
        """
        feature_indices = []
        for i in self.model_data['indices']:
            n_features = self.meta_info['n_features']
            if n_features > 1 or (n_features == 1 and i >= 0):
                feature_indices.append([str(j) for j in range(n_features)][i])

        n_indents = 1 if language in {'java', 'js', 'php', 'ruby'} else 0
        return self._create_branch(
            templates, language, converter,
            self.model_data['lefts'],
            self.model_data['rights'],
            self.model_data['thresholds'],
            self.model_data['classes'],
            feature_indices, 0, n_indents)

    def _create_branch(
            self,
            templates: dict,
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
        Parse and port a single tree estimator.

        Parameters
        ----------
        :param left_nodes : object
            The left children node.
        :param right_nodes : object
            The left children node.
        :param threshold : object
            The decision threshold.
        :param value : object
            The label or class.
        :param features : object
            The feature values.
        :param node : int
            The current node.
        :param depth : int
            The tree depth.

        Returns
        -------
        :return out : string
            The ported single tree as function or method.
        """
        out = ''
        temp_indent = templates.get('indent')
        if threshold[node] != -2.:

            out += '\n'
            temp = templates.get('if')
            temp = indent(temp, depth * temp_indent)
            val_1 = 'features[{}]'.format(features[node])
            if language == 'php':
                val_1 = '$' + val_1
            val_2 = converter(threshold[node])
            out += temp.format(val_1, '<=', val_2)

            if left_nodes[node] != -1.:
                out += self._create_branch(
                    templates, language, converter, left_nodes, right_nodes,
                    threshold, value, features, left_nodes[node], depth + 1)

            out += '\n'
            temp = templates.get('else')
            temp = indent(temp, depth * temp_indent)
            out += temp

            if right_nodes[node] != -1.:
                out += self._create_branch(
                    templates, language, converter, left_nodes, right_nodes,
                    threshold, value, features, right_nodes[node], depth + 1)

            out += '\n'
            temp = templates.get('endif')
            temp = indent(temp, depth * temp_indent)
            out += temp
        else:
            clazzes = []
            temp = 'classes[{0}] = {1}'
            if language == 'php':
                temp = '$' + temp
            temp = indent(temp, depth * temp_indent)

            for i, rate in enumerate(value[node]):
                clazz = temp.format(i, rate)
                clazz = '\n' + clazz
                clazzes.append(clazz)

            temp = templates.get('join')
            out += temp.join(clazzes) + temp
        return out
