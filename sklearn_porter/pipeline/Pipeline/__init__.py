from collections.abc import Callable
from textwrap import dedent
from typing import TYPE_CHECKING

from loguru import logger as L

# scikit-learn
from sklearn.pipeline import Pipeline as SklearnPipeline

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase

if TYPE_CHECKING:
    from sklearn_porter.Estimator import Estimator
    from sklearn_porter.Transformer import Transformer


class Pipeline(EstimatorBase, EstimatorApiABC):
    """
    Port a pipeline of transformers and a final estimator to one source file.

    The source file is the ported final estimator, into which the
    placeholders `pipeline_classes`, `pipeline_create` and
    `pipeline_transform` of its template insert the classes of the ported
    transformers, their creation and the transformation of the features of
    each sample. The templates of the transformers provide these parts as
    the blocks `class` (and `data`, the model data outside of `main`) and
    `create`, their names get the prefix of the step (e.g. `step1_`). The
    number of features can change in each step, the ported final estimator
    reads the number of features of the pipeline (`pipeline_n_features`).

    The exported model data is the model data of the final estimator with
    the additional key `transformers`, which contains the model data of each
    transformer.
    """

    SKLEARN_URL = 'sklearn.pipeline.Pipeline.html'

    SUPPORT = {
        language: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        }
        for language in enum.Language
    }

    def __init__(
        self,
        pipeline: SklearnPipeline,
        transformers: 'list[Transformer]',
        final_est: 'Estimator',
    ):
        super().__init__(pipeline)
        L.info('Create specific pipeline.')
        self.transformers = transformers
        self.final_est = final_est
        self.meta_info = final_est._estimator.meta_info
        self.DEFAULT_LANGUAGE = final_est._estimator.DEFAULT_LANGUAGE
        self.DEFAULT_TEMPLATE = final_est._estimator.DEFAULT_TEMPLATE

    def port(
        self,
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
        to_json: bool = False,
    ) -> str | tuple[str, str]:
        """
        Port the pipeline.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        class_name : str
            The class name of the ported final estimator, which is also the
            name of the generated file.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The ported pipeline (and the model data of the template `exported`).
        """
        tpls = self._load_templates(language.value.KEY)
        is_exported = template == enum.Template.EXPORTED

        # The parts of the ported transformers (with class names which
        # differ from the class name of the final estimator):
        steps, models = [], []
        names = {t.class_name for t in self.transformers} | {class_name}
        for index, transformer in enumerate(self.transformers):
            instance = f'step{index + 1}'
            tr_class_name = transformer.class_name
            if tr_class_name == class_name:
                tr_class_name = f'{class_name}{index + 1}'
                while tr_class_name in names:
                    tr_class_name += '_'
                names.add(tr_class_name)
            ported = transformer._estimator
            ported.template_globals = dict(
                _blocks=('class', 'data', 'create'),
                prefix=f'{instance}_',
                instance=instance,
                data=tpls.get_template('step_data').render(index=index),
            )
            try:
                out = ported.port(
                    language=language,
                    template=template,
                    class_name=tr_class_name,
                    converter=converter,
                    to_json=to_json,
                )
            finally:
                ported.template_globals = {}
            if isinstance(out, tuple):  # the template `exported`
                out, model = out
                models.append(model)
            blocks = {k: dedent(v).strip('\n') for k, v in dict(out).items()}
            steps.append(
                dict(instance=instance, prefix=f'{instance}_', **blocks)
            )

        # The number of features of the pipeline (of its first step), which
        # the ported final estimator reads:
        n_features = self.transformers[0]._estimator.meta_info['n_features']

        glue = set(tpls.list_templates())
        classes = [s[k] for s in steps for k in ('class', 'data') if s.get(k)]
        create = [s['create'] for s in steps if s.get('create')]
        if is_exported:
            # Read the model data of the transformers (and free it, e.g. in C):
            create.insert(0, tpls.get_template('load_data').render())
            if 'free_data' in glue:
                create.append(tpls.get_template('free_data').render())
        # The transformation of the features of each sample: in the loop over
        # the samples (`transform`, the number of features of the sample can
        # change), in the function which reads the samples (`read`, e.g. in C
        # with arrays of fixed size) or in a function which the function
        # which reads the samples calls (`chain`, e.g. in C#):
        transform = []
        if 'transform' in glue:
            transform = [
                tpls.get_template('transform').render(**step) for step in steps
            ]
        if 'read' in glue:
            classes.append(
                tpls.get_template('read').render(
                    steps=steps, n_features=n_features
                )
            )
        if 'chain' in glue:
            create.append(tpls.get_template('chain').render(steps=steps))

        # The ported final estimator with the parts of the transformers:
        ported = self.final_est._estimator
        ported.template_globals = dict(
            pipeline=True,
            pipeline_n_features=n_features,
            pipeline_classes='\n\n'.join(classes),
            pipeline_create='\n\n'.join(create),
            pipeline_transform='\n'.join(transform),
        )
        try:
            out = ported.port(
                language=language,
                template=template,
                class_name=class_name,
                converter=converter,
                to_json=to_json,
            )
        finally:
            ported.template_globals = {}

        if isinstance(out, tuple):  # the template `exported`
            out, model = out
            # The model data of the transformers is added to the object of
            # the final estimator. A list (e.g. of the trees of a forest) is
            # added as `estimators`, which the final estimator reads in a
            # pipeline (see the template variable `pipeline`):
            if model.startswith('['):
                model = f'{{"estimators":{model}}}'
            sep = ',' if model != '{}' else ''
            model = (
                f'{model[:-1]}{sep}"transformers":[{",".join(models)}]' + '}'
            )
            return out, model
        return out
