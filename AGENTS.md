# AGENTS.md

## Purpose

sklearn-porter transpiles trained scikit-learn estimators (classifiers,
regressors, outlier detectors) and transformers (e.g. for the preprocessing)
to C, C#, Go, Java, JavaScript, PHP and Ruby, e.g. for embedded systems
without Python. The ported code must predict exactly the same classes as the
original estimator and transform the features like the original transformer.

- Python `>= 3.11`, `scikit-learn >= 1.8`, managed with uv
- License: BSD 3-Clause

## Concepts

- **Templates** define how the model data is stored in the generated code:
  `attached` (data embedded as variables), `exported` (data in a separate
  JSON file), `combined` (data compiled into the code, e.g. nested if/else).
- **Methods**: `predict` and, if supported, `predict_proba` of the
  estimators, `transform` of the transformers.
- **Public API** (exported in `sklearn_porter/__init__.py`): the classes
  `Porter`, `Estimator`, `Transformer` and `Pipeline` (incl. the method
  `can`), the
  functions `port`, `save`, `make`, `test`, `can`, `show` and the dict
  `options`. The CLI `porter` (`show`, `port`, `save`) lives in
  `sklearn_porter/cli/`.
- `Porter` (`sklearn_porter/Porter.py`) is the central class (users always
  go through it): it creates a `Transformer` for a transformer
  (`is_transformer()`), a `Pipeline` for a pipeline whose preprocessing
  steps and final estimator are all supported (`Pipeline.supports()`, not
  with `ignore_preprocessing=True`), otherwise an `Estimator`, and forwards
  all calls to it. The functions of the same module and the CLI use it.
- `Estimator` (`sklearn_porter/Estimator.py`, ported classes in
  `sklearn_porter/estimator/`) and `Transformer`
  (`sklearn_porter/Transformer.py`, ported classes in
  `sklearn_porter/transformer/`) share the base class `Base`
  (`sklearn_porter/Base.py`: properties, `supports()`/`can`, `port`, `save`,
  the compilation and execution in `_run()`). `Estimator` adds the pipeline
  handling, `make()`/`predict()` and `test()`, `Transformer` adds
  `make()`/`transform()` and `test()`. Tests patch `run`, `platform` etc. in
  `sklearn_porter.Base`.
- `Pipeline` (`sklearn_porter/Pipeline.py`, also a subclass of `Base`; it
  imports scikit-learn's class as `SklearnPipeline`) validates the steps of
  a pipeline (also of a search) with `Transformer` and `Estimator` and ports
  it to one source file with `sklearn_porter/pipeline/Pipeline/` (glue
  templates per language): the ported final estimator (with the class name
  of the pipeline, by default `Pipeline`) into which the ported transformers
  are inserted. Repeated class names get a number (`StandardScaler2`).
  The transformers can change the number of features (`n_features_out` of
  `_set_model_data()`): the shared template `samples` of each language
  checks the number of features of the pipeline (template global
  `pipeline_n_features`). In C (arrays of fixed size) the glue template
  `read` defines `read_features`, which reads the features with the
  renamed shared function `pipeline_read_features` and passes them through
  the steps in separate arrays; in C# (the loop variable of `foreach` can't
  be assigned) the glue template `chain` assigns the transformations of all
  steps to `PipelineTransform`, which `ReadSamples` applies; the other
  languages assign the result of each step to `features` (glue template
  `transform`).
- Pipeline placeholders: every class template of an estimator contains the
  placeholders `pipeline_classes` (before `main`, in Java and C# after the
  class), `pipeline_create` (in `main` before the model data or the loop
  over the samples) and `pipeline_transform` (first statement in the loop),
  each wrapped in `{% if ... %}`, so the code of a single estimator doesn't
  change. The templates of a transformer provide the blocks `class` (and
  `data`, the model data outside of `main`) and `create`, which
  `EstimatorBase` renders separately if the template global `_blocks` is set
  (`template_globals`). In this mode the names get the prefix `{{ prefix }}`
  (e.g. `step1_mean`, in C also the functions and defines), the instance is
  `{{ instance }}` and the exported model data is the expression
  `{{ data }}`; parts which only a pipeline needs are wrapped in
  `{% if _blocks %}`. The exported JSON is the object of the final estimator
  with the key `transformers` (a list like the trees of a forest is under
  `estimators`, which the final estimator reads if the global `pipeline` is
  set).
- `make()` compiles and executes the ported estimator via subprocess calls,
  `test()` compares its predictions (or transformed features) with
  scikit-learn (integrity score).
- The `main` of each ported estimator reads the features of one sample from
  its arguments or of several samples (one per line) from the standard input
  (shared template `samples` of each language) and prints one result per
  line. `make()` passes the samples in batches of at least 100 samples to
  the standard input, because starting a process (e.g. the JVM) takes much
  longer than a prediction.
- The estimators print JSON for `make()` (`is_test` or `to_json`) and text
  otherwise. The transformers always print the transformed features of each
  sample in one line, separated by spaces (shared template `values` of each
  language, round-trip precision, `NaN`, `Infinity`, `-0`), which is also
  the input format of the ported estimators. The shared template `samples`
  reads `nan`, `inf` and `infinity` (with a sign, in any case) in all
  languages, i.e. the output of Python's `str` (`make()`) and of the ported
  transformers (PHP and Ruby read unknown words as `0.0`).

## Layout

- `sklearn_porter/estimator/<Name>/__init__.py`: subclass of `EstimatorBase`
  and `EstimatorApiABC`. It declares the supported combinations of language,
  template and method in `SUPPORT`, extracts `meta_info` and `model_data` from
  the fitted estimator and renders the templates in `port()`.
- `sklearn_porter/estimator/<Name>/templates/<language>/*.jinja2`: Jinja2
  templates; they extend the base templates in
  `sklearn_porter/language/<language>/templates/`. A subclass of another
  estimator class (e.g. `GradientBoostingClassifier` of
  `GradientBoostingRegressor` of `RandomForestRegressor`) loads the
  templates of all its base classes, and its own templates overwrite the ones
  with the same name.
- `sklearn_porter/transformer/<Name>/__init__.py` and
  `templates/<language>/*.jinja2`: the same for transformers, subclass of
  `TransformerBase` (of `EstimatorBase`), registered in
  `Transformer.transformers()` in `sklearn_porter/Transformer.py`. The
  templates are loaded from the directory of the module of each class.
  `TransformerBase.port()` renders the model data, which consists of finite
  arrays, mostly with one number per feature or per transformed feature
  (`_set_model_data()`, e.g. `mean` and `scale`, with `n_features_out` for
  transformers which change the number of features like the
  `OneHotEncoder`; arrays of integers like the offsets of the
  concatenated categories of the `OrdinalEncoder` are passed as `ints`), as
  arrays (`attached`), numbers in the code (`combined`) or JSON
  (`exported`); parameters which are numbers in the code of all templates
  (e.g. the bounds of `clip=True` of the `MinMaxScaler`) come from
  `literals()`, whose converter writes `NaN` and infinite values as
  literals of the language (templates `nan` and `inf`). Go may fuse `a * b + c` to an FMA instruction, so the
  templates write `float64(a*b) + c` to keep the results identical.
- Composite transformers: `sklearn_porter/transformer/ColumnTransformer/`
  (and its subclass `FeatureUnion`) ports the transformers of each part (a
  transformer, the steps of a pipeline or none for `'passthrough'`) in the
  block mode of a pipeline (prefix `t<part>_<step>_`, instance
  `<instance>_t<part>_<step>`, exported model data under `transformers`)
  and renders its own templates (`class`, `create`, `program` for a single
  composite transformer, `load_data`, `root_data`, `child_data`). Its class
  holds the instances of the transformers (in C a function with arrays per
  part). `Transformer.supports()` checks the transformers of all parts.
- Feature selectors: `sklearn_porter/transformer/SelectorMixin/` (not
  registered) extracts the indices of the selected features
  (`get_support()`) and holds the templates; each selector (e.g.
  `SelectKBest`, `RFE`) is a subclass in its own module, because the
  registry loads the module by the name of the class of scikit-learn.
- `sklearn_porter/language/<language>/__init__.py`: subclass of `LanguageABC`
  with the compile and execute commands and small code snippets.
- `sklearn_porter/enums`, `exceptions`, `utils`, `meta`: shared helpers.
- `tests/`: `test_estimator.py` (API, CLI, regression of all classifiers),
  `tests/estimator/test_*.py` (parameter sweeps per estimator),
  `test_transformer.py` and `tests/transformer/test_*.py` (the same for
  transformers), `test_pipeline.py` and `tests/pipeline/test_pipeline.py`
  (pipelines), `test_porter.py` (the central class), fixtures in
  `tests/conftest.py`.
- `README.md` is the user documentation (also shown on PyPI, so use absolute
  links there), `CONTRIBUTING.md` the developer documentation.
- `docs/`: further user documentation linked from the README
  (`estimator-notes.md`, `limitations.md`).
- `examples/basics/index.pct.py`: example notebook in jupytext format; the
  `.ipynb` file is generated.

## Commands

```bash
uv sync --all-groups   # install all dependencies
make lint              # ruff check + ruff format --check + ty check
make format            # fix and format with ruff
make test              # fast tests on the host (uv run pytest)
make test-regression   # regression tests in Docker (slow, ~20+ min)
```

## Testing

- Tests which compile and execute ported estimators are marked with
  `@pytest.mark.regression` and skipped by default
  (`addopts = "-m 'not regression'"` in `pyproject.toml`).
- The regression tests need gcc, dotnet, go, java, node, php and ruby. They
  run only locally in Docker (`make test-regression`), never in GitHub
  Actions, because they would exceed the free minutes.
- The tests are parametrized with the keys of all languages from
  `tests.commons.LANGUAGES`.
- Run the regression tests in Docker after changing templates, the model data
  extraction or the execution of ported estimators.
- New tests which call `make()` or `test()` must be marked with `regression`.
- Don't compare floating point results with NumPy bit-exactly (e.g.
  `assert_array_equal` of scores or probabilities): NumPy computes functions
  like `np.power`, `np.exp` or `np.log` with SIMD instructions depending on
  the CPU (e.g. SVML with AVX-512 on the x86 runners of GitHub Actions),
  which can differ by a few ULP from `pow()`/`exp()` of libm or of the target
  languages. Docker on Apple Silicon (aarch64) doesn't show these differences,
  so a test can pass locally and fail on some runners only. Use
  `assert_allclose` with a small `rtol` (e.g. `1e-14`) and drop samples on a
  decision boundary (ties) before comparing predictions.

## Adding or changing an estimator

1. Implement or update `sklearn_porter/estimator/<Name>/__init__.py`,
   including `SUPPORT`.
2. Add or update the Jinja2 templates for every supported language and
   template, including the pipeline placeholders (see above).
3. Register the estimator in `Estimator.classifiers()`,
   `Estimator.regressors()` or `Estimator.outlier_detectors()` in
   `sklearn_porter/Estimator.py`.
4. Add tests and run the regression tests in Docker.
5. Add the estimator to the manually maintained list of classifiers,
   regressors or outlier detectors in `README.md`.

Goal: every estimator supports all languages, templates and methods
(`predict_proba` only for classifiers), so the README needs no support
table.

## Adding or changing a transformer

1. Implement or update `sklearn_porter/transformer/<Name>/__init__.py`
   (subclass of `TransformerBase`), including `SUPPORT` with
   `Method.TRANSFORM`; extract the model data with `_get_n_features()` and
   `_set_model_data()`.
2. Add or update the Jinja2 templates for every supported language and
   template; `main` prints the result of `transform` with the template
   `values`. Provide the blocks `class` (and `data`) and `create` for
   pipelines (see above).
3. Register the transformer in `Transformer.transformers()` in
   `sklearn_porter/Transformer.py`.
4. Add tests (fast ones in `tests/test_transformer.py`, regression tests in
   `tests/transformer/` and as step of a pipeline in `tests/pipeline/`) and
   run the regression tests in Docker.
5. Add the transformer to the list of transformers in `README.md`.

## scikit-learn specifics

- Use only public scikit-learn imports (e.g. `from sklearn.tree import ...`).
- Since scikit-learn 1.4 `tree_.value` contains the (weighted) class
  fractions; the tree templates use them directly as floating point numbers
  (`sklearn_porter.utils.get_class_fractions()`). Don't round them to class
  counts, otherwise the probabilities of trees fitted with `sample_weight` or
  `class_weight` differ.
- The model data of the tree estimators contains only the splits of each tree
  (`sklearn_porter.utils.get_tree_data()`): the children refer to splits by
  `i >= 0` and to leaves by `-i - 1`, the values of the leaves are
  concatenated (the values of leaf `i` start at `i * n_values`). A tree with
  a single leaf gets a split whose children are both this leaf, so no array
  is empty.
- scikit-learn routes missing values (NaN) of every tree via
  `tree_.missing_go_to_left`, even if the tree was fitted without missing
  values. The tree templates check `isnan` only for splits which send missing
  values to the left child, because all comparisons with NaN are false.
- Trees fitted on data with missing values can have infinite thresholds. The
  tree estimators write them as literals of the language (`with_infinity()`,
  template `inf`) and as `1e999` in JSON (`dump_model_data()`); Go parses them
  with a custom `UnmarshalJSON` (`has_infinite_thresholds`).
- `Estimator` converts the sparse matrices of a fitted estimator (e.g. after a
  fit on sparse data or `sparsify()`) to dense arrays before the model data is
  extracted (`sklearn_porter.utils.densify()`), so the estimator classes don't
  need to check for sparse matrices themselves. The passed estimator remains
  unchanged and is used for the predictions of scikit-learn.
- The ported classifiers return the labels of the classes
  (`sklearn_porter.utils.get_labels()`): `predict_index` returns the index
  of the class and `predict` maps it to the labels, which the shared
  template `labels` of each language
  (`sklearn_porter/language/<language>/templates/labels.jinja2`) defines.
  All labels have the same kind (`int`, `double` or `str`), the string
  literals are escaped by `str_literal()` of the language. Multilabel
  classifiers keep their `predict` (one binary value per label) without
  `predict_index`.
- AdaBoostClassifier is ported with the SAMME algorithm (`SAMME.R` was removed
  from scikit-learn).

## Conventions

- Format and lint with ruff (line length 80, single quotes).
- Type check with ty (`uvx ty check`); use `# ty: ignore[<rule>]` only for
  attributes which scikit-learn sets dynamically.
- NumPy style docstrings and type hints.
- PHP templates declare all properties of their classes explicitly, because
  dynamic properties are deprecated since PHP 8.2. The regression tests run
  PHP with `error_reporting = E_ALL` (see `tests/conftest.py`), so any notice,
  warning or deprecation breaks the JSON output and fails the tests.
- C# (key `cs`): `make()` compiles the source code directly with the
  compiler of the .NET SDK (`csc.dll`, much faster than `dotnet build`) and
  `-langversion:7.3`, so the templates must only use language features of
  C# 7.3. The compiler gets the reference assemblies from a response file
  (`*.rsp`) and the program a runtime configuration
  (`*.runtimeconfig.json`), see `CSharp.prepare_compilation()`. The
  `exported` templates read the JSON file with `System.Text.Json` and
  `JsonNamingPolicy.SnakeCaseLower` (.NET 8+). Format and parse numbers with
  `CultureInfo.InvariantCulture`, because the default culture can use a
  comma as decimal separator. `Array.Sort` isn't stable, so sort by an
  additional index where the order of equal values matters (e.g. the
  neighbors of the KNeighborsClassifier).
- Commit messages and PR titles follow Conventional Commits
  (`feat:`, `fix:`, `docs:`, `ci:`, `build:`, `style:`, `refactor:`,
  `test:`, `chore:`; `!` or `BREAKING CHANGE:` for breaking changes).
- `main` is protected: all changes go through pull requests which are squash
  merged, so the PR title becomes the commit message.

## Releases

- release-please (`.github/workflows/release.yml`) opens a release PR on
  `main` that bumps the version in `pyproject.toml` and `uv.lock` and updates
  `CHANGELOG.md`. Don't change the version or the changelog manually.
- Merging the release PR creates the tag `vX.Y.Z` and the GitHub release,
  then the package is published to TestPyPI, verified and published to PyPI
  with trusted publishing.
