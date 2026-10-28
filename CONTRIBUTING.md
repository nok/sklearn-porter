# Contributing

Thanks for your interest in sklearn-porter! This guide describes how to set up the project, run the tests and get your changes merged.


## Setup

The Python version and all dependencies are managed with [uv](https://docs.astral.sh/uv/). Install everything with:

```bash
uv sync --all-groups
```


## Commands

```bash
make lint             # Check the code with ruff and ty
make format           # Fix and format the code with ruff
make test             # Run the fast tests
make test-regression  # Run the regression tests in Docker
make test-all         # Run all tests with coverage in Docker
make notebook         # Start Jupyter with the examples
make examples         # Generate the example notebooks with jupytext
```

The code is formatted and linted with [ruff](https://docs.astral.sh/ruff/) and type checked with [ty](https://docs.astral.sh/ty/).


## Testing

The fast tests check the Python code without running the ported estimators. GitHub Actions runs them on every push and pull request with Python 3.11 to 3.14 ([workflow](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml)):

```bash
uv run pytest
```

The regression tests compile and run the ported estimators and compare their predictions with the original estimators. They take a long time (20+ minutes) and need all compilers and interpreters. Therefore they are marked with `regression`, skipped by default and only run locally in [Docker](Dockerfile):

```bash
make test-regression
```

Run them after changing templates, the extraction of the model data or the execution of the ported estimators. New tests which call `make` or `test` must be marked with `@pytest.mark.regression`.

The tests use the Python version from [`.python-version`](.python-version). Set `PYTHON_VERSION` to use another one:

```bash
make test-regression PYTHON_VERSION=3.14
```

The Docker image (Debian 12 "bookworm") contains these compilers and interpreters:

| Language   | Compiler / Interpreter |
|------------|------------------------|
| C          | GCC 12                 |
| C#         | .NET SDK 10            |
| Go         | Go 1.19                |
| Java       | OpenJDK 17             |
| JavaScript | Node.js 18             |
| PHP        | PHP 8.2                |
| Ruby       | Ruby 3.1               |

In general, the generated source code works with older compilers and interpreters too. For instance, you can use Java 8 to compile and run the models. The .NET SDK isn't available as Debian package for all architectures, so the [`Dockerfile`](Dockerfile) copies it from the official image `mcr.microsoft.com/dotnet/sdk`.


## Adding or changing an estimator

1. Implement or update `sklearn_porter/estimator/<Name>/__init__.py`, including the supported combinations in `SUPPORT`.
2. Add or update the Jinja2 templates in `sklearn_porter/estimator/<Name>/templates/<language>/` for every supported language and template, including the placeholders `pipeline_classes`, `pipeline_create` and `pipeline_transform`, which insert the transformers of a pipeline (see `AGENTS.md`).
3. Register the estimator in `Estimator.classifiers()`, `Estimator.regressors()` or `Estimator.outlier_detectors()` in [`sklearn_porter/Estimator.py`](sklearn_porter/Estimator.py).
4. Add tests and run the regression tests in Docker.
5. Add the estimator to the list of classifiers, regressors or outlier detectors in the [README](README.md).

The goal is that every estimator supports all languages, templates and methods (`predict_proba` only for classifiers).


## Adding or changing a transformer

1. Implement or update `sklearn_porter/transformer/<Name>/__init__.py` as a subclass of `TransformerBase`, including the supported combinations in `SUPPORT` (with the method `transform`).
2. Add or update the Jinja2 templates in `sklearn_porter/transformer/<Name>/templates/<language>/` for every supported language and template. The method `transform` returns the transformed features of one sample, the `main` prints them with the shared template `values` of each language (one sample per line, separated by spaces). The blocks `class` (and `data`) and `create` insert the transformer into a ported pipeline (see `AGENTS.md`).
3. Register the transformer in `Transformer.transformers()` in [`sklearn_porter/Transformer.py`](sklearn_porter/Transformer.py).
4. Add tests in `tests/transformer/` and run the regression tests in Docker.
5. Add the transformer to the list of transformers in the [README](README.md).


## Pull requests

Commit messages and pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/), e.g. `feat: add support for ...` or `fix(php): ...`. Pull requests are squash merged into `main`, so the pull request title becomes the commit message.


## Releases

Releases are created with [release-please](https://github.com/googleapis/release-please-action):

1. release-please collects the Conventional Commits on `main` in a release pull request. It updates the version and the [changelog](CHANGELOG.md).
2. Merging the release pull request creates the Git tag and the GitHub release.
3. The package is published to [TestPyPI](https://test.pypi.org/project/sklearn-porter/), installed and checked from there and only then published to [PyPI](https://pypi.org/project/sklearn-porter/).
