# AGENTS.md

## Purpose

sklearn-porter transpiles trained scikit-learn estimators to C, Go, Java,
JavaScript, PHP and Ruby, e.g. for embedded systems without Python. The ported
code must predict exactly the same classes as the original estimator.

- Python `>= 3.11`, `scikit-learn >= 1.8`, managed with uv
- License: BSD 3-Clause

## Concepts

- **Templates** define how the model data is stored in the generated code:
  `attached` (data embedded as variables), `exported` (data in a separate
  JSON file), `combined` (data compiled into the code, e.g. nested if/else).
- **Methods**: `predict` and, if supported, `predict_proba`.
- **Public API** (exported in `sklearn_porter/__init__.py`): class
  `Estimator` (incl. the method `can`), the functions `port`, `save`, `make`,
  `test`, `show` and the dict `options`. The CLI `porter` (`show`, `port`,
  `save`) lives in `sklearn_porter/cli/`.
- `make()` compiles and executes the ported estimator via subprocess calls,
  `test()` compares its predictions with scikit-learn (integrity score).

## Layout

- `sklearn_porter/estimator/<Name>/__init__.py`: subclass of `EstimatorBase`
  and `EstimatorApiABC`. It declares the supported combinations of language,
  template and method in `SUPPORT`, extracts `meta_info` and `model_data` from
  the fitted estimator and renders the templates in `port()`.
- `sklearn_porter/estimator/<Name>/templates/<language>/*.jinja2`: Jinja2
  templates; they extend the base templates in
  `sklearn_porter/language/<language>/templates/`.
- `sklearn_porter/language/<language>/__init__.py`: subclass of `LanguageABC`
  with the compile and execute commands and small code snippets.
- `sklearn_porter/enums`, `exceptions`, `utils`, `meta`: shared helpers.
- `tests/`: `test_estimator.py` (API, CLI, regression of all classifiers),
  `tests/estimator/test_*.py` (parameter sweeps per estimator), fixtures in
  `tests/conftest.py`.
- `README.md` is the user documentation (also shown on PyPI, so use absolute
  links there), `CONTRIBUTING.md` the developer documentation.
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
- The regression tests need gcc, go, java, node, php and ruby. They run only
  locally in Docker (`make test-regression`), never in GitHub Actions, because
  they would exceed the free minutes.
- Run the regression tests in Docker after changing templates, the model data
  extraction or the execution of ported estimators.
- New tests which call `make()` or `test()` must be marked with `regression`.

## Adding or changing an estimator

1. Implement or update `sklearn_porter/estimator/<Name>/__init__.py`,
   including `SUPPORT`.
2. Add or update the Jinja2 templates for every supported language and
   template.
3. Register the estimator in `Estimator.classifiers()` or
   `Estimator.regressors()` in `sklearn_porter/Estimator.py`.
4. Add tests and run the regression tests in Docker.
5. Add the estimator to the manually maintained list of classifiers or
   regressors in `README.md`.

Goal: every estimator supports all languages, templates and methods
(`predict_proba` only for classifiers), so the README needs no support
table.

## scikit-learn specifics

- Use only public scikit-learn imports (e.g. `from sklearn.tree import ...`).
- Since scikit-learn 1.4 `tree_.value` contains the (weighted) class
  fractions; the tree templates use them directly as floating point numbers
  (`sklearn_porter.utils.get_class_fractions()`). Don't round them to class
  counts, otherwise the probabilities of trees fitted with `sample_weight` or
  `class_weight` differ.
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
