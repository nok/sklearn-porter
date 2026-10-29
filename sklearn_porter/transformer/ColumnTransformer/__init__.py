from collections.abc import Callable
from copy import deepcopy
from textwrap import dedent
from typing import Any, cast

from loguru import logger as L

# scikit-learn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer as ColumnTransformerClass
from sklearn.pipeline import Pipeline as SklearnPipeline

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


def split_steps(transformer: object) -> list:
    """
    Get the transformers of a part of a composite transformer.

    A pipeline of transformers is split into its steps (without the skipped
    ones), `'passthrough'` has no steps.

    Parameters
    ----------
    transformer : object
        The transformer of the part (e.g. a `StandardScaler`) or a string
        (e.g. `'passthrough'`).

    Returns
    -------
    The transformers, which are applied one after another.
    """
    if isinstance(transformer, str):
        return []
    if isinstance(transformer, SklearnPipeline):
        return [
            step
            for _, step in transformer.steps
            if step is not None and not isinstance(step, str)
        ]
    return [transformer]


def get_parts(transformer: BaseEstimator) -> list[tuple[str, object]]:
    """
    Get the parts (name and transformer) of a composite transformer.

    Parameters
    ----------
    transformer : BaseEstimator
        A `ColumnTransformer` or a `FeatureUnion` (fitted or not).

    Returns
    -------
    The names and transformers of the parts (also the dropped ones).
    """
    if isinstance(transformer, ColumnTransformerClass):
        parts = getattr(transformer, 'transformers_', transformer.transformers)
        return [(name, trans) for name, trans, _ in parts]
    return list(getattr(transformer, 'transformer_list', []))


class ColumnTransformer(TransformerBase, EstimatorApiABC):
    """
    Extract the parts and port a ColumnTransformer transformer.

    Each part transforms the selected features (columns) with its
    transformers (a transformer, the steps of a pipeline or none with
    `'passthrough'`), the transformed features of all parts are
    concatenated (and multiplied by the weights of the parts). The ported
    transformers of the parts are inserted like the steps of a pipeline:
    their names get the prefix of the part (e.g. `t1_1_`), the ported class
    of the column transformer holds their instances.
    """

    SKLEARN_URL = 'sklearn.compose.ColumnTransformer.html'

    DEFAULT_LANGUAGE = enum.Language.JS
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.TRANSFORM

    SUPPORT = {
        language: {
            enum.Template.ATTACHED: {enum.Method.TRANSFORM},
            enum.Template.COMBINED: {enum.Method.TRANSFORM},
            enum.Template.EXPORTED: {enum.Method.TRANSFORM},
        }
        for language in enum.Language
    }

    def __init__(self, transformer: BaseEstimator):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        n_features = self._get_n_features(transformer)
        self.parts = self._get_parts(transformer, n_features)
        n_features_out = sum(part['n_out'] for part in self.parts)
        self._set_model_data(n_features, n_features_out=n_features_out)

    def _get_parts(self, transformer: BaseEstimator, n_features: int) -> list:
        """
        Get the parts with their columns, transformers and weights.

        Parameters
        ----------
        transformer : BaseEstimator
            The fitted column transformer.
        n_features : int
            The number of features.

        Returns
        -------
        The parts in the order of their transformed features.
        """
        est: Any = transformer  # alias (with attributes of scikit-learn)
        # The positions of the selected columns of each part (also of
        # columns selected by names or callables) and the positions of
        # their transformed features:
        indices = est._transformer_to_input_indices
        outputs = est.output_indices_
        weights = est.transformer_weights or {}
        parts = []
        for name, trans, _ in est.transformers_:
            columns = [int(i) for i in indices.get(name, [])]
            if isinstance(trans, str) or not columns:  # e.g. `'drop'`
                continue
            part = self._create_part(name, trans, columns, weights.get(name))
            output = outputs[name]
            if output.stop - output.start != part['n_out']:
                msg = (
                    f'The number of transformed features of the part `{name}` '
                    f'of the transformer `{self.estimator_name}` is unexpected.'
                )
                raise exception.NotSupportedYetError(msg)
            part['offset'] = output.start
            parts.append(part)
        return sorted(parts, key=lambda part: part['offset'])

    def _create_part(
        self,
        name: str,
        transformer: object,
        columns: list[int],
        weight: float | None,
    ) -> dict:
        """Create the ported transformers of a part."""
        from sklearn_porter.Transformer import Transformer

        steps = []
        for step in split_steps(transformer):
            if not Transformer.supports(step):
                msg = (
                    f'The part `{name}` of the transformer '
                    f'`{self.estimator_name}` contains the transformer '
                    f'`{type(step).__name__}`, which is not supported yet.'
                )
                raise exception.NotSupportedYetError(msg)
            steps.append(Transformer(step))
        n_out = len(columns)
        if steps:
            n_out = steps[-1]._estimator.meta_info['n_features_out']
        return dict(
            name=name,
            columns=columns,
            steps=steps,
            weight=None if weight is None else float(weight),
            n_out=n_out,
        )

    def port(
        self,
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
        to_json: bool = False,
    ) -> str | tuple[str, str]:
        """
        Port the composite transformer with the transformers of its parts.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        class_name : str
            The class name of the ported transformer.
        converter : Callable
            The converter of the floating numbers.
        to_json : bool (default: False)
            Unused, because the ported transformers always print the
            transformed features of each sample in one line.

        Returns
        -------
        The ported transformer (and the model data of the template
        `exported`), or as step of a pipeline its blocks.
        """
        tpls = self._load_templates(language.value.KEY)
        is_exported = template == enum.Template.EXPORTED
        is_step = bool(self.template_globals.get('_blocks'))
        prefix = self.template_globals.get('prefix', '')
        instance = self.template_globals.get('instance', 'transformer')
        if is_step:
            parent = self.template_globals.get('data', '')
        else:
            parent = tpls.get_template('root_data').render()

        # The parts of the ported transformers of all parts:
        classes, data, create, models, parts = [], [], [], [], []
        names = set()
        index = 0  # of the model data of the transformers of all parts
        for i, part in enumerate(self.parts, start=1):
            steps = []
            for j, step in enumerate(part['steps'], start=1):
                step_class_name = f'{class_name}{step.class_name}'
                while step_class_name in names:
                    step_class_name += '_'
                names.add(step_class_name)
                ported = step._estimator
                ported.template_globals = dict(
                    _blocks=('class', 'data', 'create'),
                    prefix=f'{prefix}t{i}_{j}_',
                    instance=f'{instance}_t{i}_{j}',
                    data=tpls.get_template('child_data').render(
                        parent=parent, index=index, prefix=prefix
                    ),
                )
                try:
                    out = ported.port(
                        language=language,
                        template=template,
                        class_name=step_class_name,
                        converter=converter,
                        to_json=to_json,
                    )
                finally:
                    ported.template_globals = {}
                if isinstance(out, tuple):  # the template `exported`
                    out, model = out
                    models.append(model)
                blocks = {
                    k: dedent(v).strip('\n') for k, v in dict(out).items()
                }
                classes.append(blocks.get('class', ''))
                data.append(blocks.get('data', ''))
                create.append(blocks.get('create', ''))
                steps.append(
                    dict(
                        class_name=step_class_name,
                        prefix=f'{prefix}t{i}_{j}_',
                        instance=f'{instance}_t{i}_{j}',
                        field=f't{i}_{j}',
                        n_out=ported.meta_info['n_features_out'],
                    )
                )
                index += 1
            weight = part['weight']
            parts.append(
                dict(
                    columns=part['columns'],
                    steps=steps,
                    n_out=part['n_out'],
                    offset=part['offset'],
                    weight=None if weight is None else converter(weight),
                )
            )

        plas = deepcopy(self.placeholders)
        plas.update(self.meta_info)
        plas.update(
            dict(
                class_name=class_name,
                prefix=prefix,
                instance=instance,
                parts=parts,
                steps=[step for part in parts for step in part['steps']],
                is_exported=is_exported,
                is_step=is_step,
            )
        )
        own_create = tpls.get_template('create').render(**plas)
        if is_step:
            own_class = tpls.get_template('class').render(**plas)
            # The blocks of a step of a pipeline (like `_Template.render`):
            blocks = dict(
                {'class': _join(*classes, own_class)},
                data=_join(*data),
                create=_join(*create, own_create),
            )
            out = cast(str, blocks)
        else:
            load = free = ''
            if is_exported:
                load = tpls.get_template('load_data').render()
                if 'free_data' in tpls.list_templates():
                    free = tpls.get_template('free_data').render()
            plas['create'] = _join(load, *create, free, own_create)
            own_class = tpls.get_template('class').render(**plas)
            plas['classes'] = _join(*classes, *data, own_class)
            plas['template'] = template.value
            out = tpls.get_template('program').render(**plas)
        if is_exported:
            return out, '{"transformers":[' + ','.join(models) + ']}'
        return out


def _join(*parts: str) -> str:
    """Join the non-empty parts of the code with an empty line."""
    return '\n\n'.join(part for part in parts if part.strip())
