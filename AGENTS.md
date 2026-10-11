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
- **Public API** (`sklearn_porter/Estimator.py`): class `Estimator` and the
  functions `port`, `save`, `make`, `test`, `show`, `can`. The CLI `porter`
  (`show`, `port`, `save`) lives in `sklearn_porter/cli/`.
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
- `examples/basics/index.pct.py`: example notebook in jupytext format; the
  `.ipynb` file is generated.

## Commands

```bash
uv sync --all-groups   # install all dependencies
make lint              # ruff check + ruff format --check
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
5. Update the support table in `readme.md`.

## scikit-learn specifics

- Use only public scikit-learn imports (e.g. `from sklearn.tree import ...`).
- Since scikit-learn 1.4 `tree_.value` contains class fractions; use
  `sklearn_porter.utils.get_class_counts()` to get the integer class counts
  which the tree templates expect.
- AdaBoostClassifier is ported with the SAMME algorithm (`SAMME.R` was removed
  from scikit-learn).

## Conventions

- Format and lint with ruff (line length 80, single quotes).
- NumPy style docstrings and type hints.
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
