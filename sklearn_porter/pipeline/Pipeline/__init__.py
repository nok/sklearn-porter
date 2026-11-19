from collections.abc import Callable
from textwrap import dedent
from typing import TYPE_CHECKING

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

    The source file contains the ported final estimator (with its own class
    name, e.g. `DecisionTreeClassifier`), the ported transformers and the
    class of the pipeline (glue template `class`, with the class name of the
    pipeline), which holds the transformers and the final estimator: its
    factories create them (`create()` or `load(path)`), `transform` applies
    the transformers and the methods of the final estimator (e.g. `predict`)
    predict the transformed features. The placeholders `pipeline_classes`,
    `pipeline_create` and `pipeline_transform` of the template of the final
    estimator insert the classes, the creation of the pipeline and the
    transformation of the features of each sample into the program (`main`
    of the final estimator). The templates of the transformers provide their
    parts as the blocks `class` (and `data`, the model data outside of
    `main`) and `create`, their names get the prefix of the step (e.g.
    `step1_`). The number of features can change in each step, the ported
    final estimator reads the number of features of the pipeline
    (`pipeline_n_features`).

    In C (without classes) functions with the prefix `pipeline_` transform
    the features (`pipeline_transform`) and predict the transformed features
    (e.g. `pipeline_predict`). In JavaScript the class of the final estimator
    has the class name of the pipeline and a function creates the
    transformers (`createSteps`), which the program and the pipeline as
    module (`createPipeline`, e.g. in a browser) use.

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
        glue = set(tpls.list_templates())

        def render(name: str, **kwargs) -> str:
            """Render a glue template without surrounding blank lines."""
            return tpls.get_template(name).render(**kwargs).strip('\n')

        is_exported = template == enum.Template.EXPORTED

        # The class name of the ported final estimator, which differs from
        # the class names of the pipeline and of the transformers (only
        # with a class of the pipeline, see the glue template `class`):
        names = {t.class_name for t in self.transformers} | {class_name}
        est_class_name = class_name
        if 'class' in glue:
            est_class_name, i = self.final_est.class_name, 1
            while est_class_name in names:
                i += 1
                est_class_name = f'{self.final_est.class_name}{i}'
            names.add(est_class_name)

        # The parts of the ported transformers (with class names which
        # differ from the class name of the final estimator):
        steps, models = [], []
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
                data=render('step_data', index=index),
            )
            if 'on_error' in glue:  # e.g. Go: return errors in the factory
                ported.template_globals['on_error'] = render(
                    'on_error', class_name=class_name
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
                dict(
                    instance=instance,
                    prefix=f'{instance}_',
                    class_name=tr_class_name,
                    **blocks,
                )
            )

        # The number of features of the pipeline (of its first step), which
        # the ported final estimator reads:
        n_features = self.transformers[0]._estimator.meta_info['n_features']

        classes = [s[k] for s in steps for k in ('class', 'data') if s.get(k)]
        create = [s['create'] for s in steps if s.get('create')]
        if 'steps' in glue:
            # A function creates the transformers (`steps`, e.g. in
            # JavaScript), which the program and the pipeline as module
            # (e.g. in a browser) use:
            classes.append(
                render(
                    'steps',
                    steps=steps,
                    is_exported=is_exported,
                    template=template.value,
                    class_name=class_name,
                )
            )
            create = [render('create', is_exported=is_exported)]
            transform = [render('transform')]
        elif 'class' in glue:
            # The class of the pipeline creates the transformers, which the
            # program uses:
            create = [
                render('create', class_name=class_name, is_exported=is_exported)
            ]
            transform = []
            if 'transform' in glue:
                transform = [render('transform')]
        else:
            # Without classes (e.g. in C) `main` creates the transformers
            # (and reads their model data), the functions of the pipeline
            # (`functions`) transform the features of each sample, which
            # they read:
            if is_exported:
                create = [
                    render('load_data'),
                    *create,
                    render('free_data'),
                ]
            transform = []

        # The ported final estimator with the parts of the transformers:
        ported = self.final_est._estimator
        template_globals = dict(
            pipeline=True,
            pipeline_class_name=class_name,
            pipeline_n_features=n_features,
            pipeline_create='\n\n'.join(create),
            pipeline_transform='\n'.join(transform),
        )
        if 'class' in glue or 'functions' in glue:
            # The methods of the final estimator, which the class of the
            # pipeline (or the functions in C) forward:
            ported.template_globals = dict(
                template_globals, pipeline_classes=''
            )
            try:
                out = ported.port(
                    language=language,
                    template=template,
                    class_name=est_class_name,
                    converter=converter,
                )
            finally:
                ported.template_globals = {}
            source = out[0] if isinstance(out, tuple) else out
            methods = language.value.methods(source, est_class_name)
            name = 'class' if 'class' in glue else 'functions'
            classes.append(
                render(
                    name,
                    steps=steps,
                    methods=methods,
                    class_name=class_name,
                    estimator_class_name=est_class_name,
                    is_exported=is_exported,
                    template=template.value,
                    n_features=n_features,
                    load_data=render(
                        'load_data', class_name=class_name, path='path'
                    )
                    if is_exported
                    else '',
                )
            )
        ported.template_globals = dict(
            template_globals, pipeline_classes='\n\n'.join(classes)
        )
        try:
            out = ported.port(
                language=language,
                template=template,
                class_name=est_class_name,
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
