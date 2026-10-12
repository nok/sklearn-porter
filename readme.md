
# sklearn-porter

[![CI](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml/badge.svg)](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/sklearn-porter.svg?color=blue)](https://pypi.org/project/sklearn-porter/)
[![Python versions](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.org/project/sklearn-porter/)
[![License](https://img.shields.io/pypi/l/sklearn-porter.svg?color=blue)](https://github.com/nok/sklearn-porter/blob/main/LICENSE)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators to C, Go, Java, JavaScript, PHP and Ruby.

Use the generated code to make predictions where Python isn't available or too heavy, e.g. on embedded systems, in web browsers or in existing applications. The ported estimators make the same predictions as the original estimators of scikit-learn.

Navigation: [Estimators](#estimators) • [Installation](#installation) • [Usage](#usage) • [Known Issues](#known-issues) • [Development](#development) • [Citation](#citation) • [License](#license)


## Estimators

The table shows the supported estimators for each programming language and template.

<table>
  <tr>
    <th rowspan="2"></th>
    <th colspan="18">Programming language</th>
  </tr>
  <tr align="center">
    <th colspan="3">C</th>
    <th colspan="3">Go</th>
    <th colspan="3">Java</th>
    <th colspan="3">JS</th>
    <th colspan="3">PHP</th>
    <th colspan="3">Ruby</th>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">svm.SVC</a>
    </td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html">svm.NuSVC</a>
    </td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">svm.LinearSVC</a>
    </td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">tree.DecisionTreeClassifier</a>
    </td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">ensemble.RandomForestClassifier</a>
    </td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">ensemble.ExtraTreesClassifier</a>
    </td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">ensemble.AdaBoostClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">neighbors.KNeighborsClassifier</a>
    </td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB">naive_bayes.BernoulliNB</a>
    </td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB">naive_bayes.GaussianNB</a>
    </td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">neural_network.MLPClassifier</a>
    </td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html">neural_network.MLPRegressor</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td rowspan="2"></td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
  </tr>
  <tr>
    <th colspan="18">Template</th>
  </tr>
</table>

✓ = `predict` is supported,　ᴾ = `predict_proba` is supported too,　× = not supported<br>
ᴀ = attached: the model data is part of the code,　ᴇ = exported: the model data is saved in a separate JSON file,　ᴄ = combined: the model data is compiled into the logic of the code (e.g. nested if-else statements)


## Installation

```bash
# with uv:
uv add sklearn-porter

# or with pip:
pip install sklearn-porter
```

sklearn-porter requires Python 3.11 or newer and scikit-learn 1.8 or newer. It's tested with Python 3.11, 3.12, 3.13 and 3.14. For older versions of Python and scikit-learn use [sklearn-porter 0.7.4](https://pypi.org/project/sklearn-porter/0.7.4/).

You only need a compiler or interpreter of the target language (e.g. `node` for JavaScript) if you want to run the ported estimator with `make` or `test`. `port` and `save` work without it.


## Usage

### Basics

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter import make, port, save, test

# 1. Load the data and train a classifier:
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)

# 2. Port (transpile) the classifier and get the source code:
output = port(clf, language='js', template='attached')
print(output)

# 3. Save the ported classifier to files:
src_path, json_path = save(
    clf, language='js', template='exported', directory='/tmp'
)
print(src_path, json_path)

# 4. Make predictions with the ported classifier:
y_classes, y_probas = make(clf, X[:10], language='js', template='exported')
print(y_classes, y_probas)

# 5. Check the ported classifier: the score is the share of predictions
#    which are equal to the predictions of the original classifier.
score = test(clf, X[:10], language='js', template='exported')
print(score)
```

### Object-oriented

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter import Estimator

# 1. Load the data and train a classifier:
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)

# 2. Port (transpile) the classifier and get the source code:
est = Estimator(clf, language='js', template='attached')
output = est.port()
print(output)

# 3. Save the ported classifier to files:
est.template = 'exported'
src_path, json_path = est.save(directory='/tmp')
print(src_path, json_path)

# 4. Make predictions with the ported classifier:
y_classes, y_probas = est.make(X[:10])
print(y_classes, y_probas)

# 5. Check the ported classifier: the score is the share of predictions
#    which are equal to the predictions of the original classifier.
score = est.test(X[:10])
print(score)
```

### Command line

After the installation the command `porter` is available:

```
porter {show,port,save} [-h] [-v]

porter show [-l {c,go,java,js,php,ruby}] [-h]

porter port <estimator> [-l {c,go,java,js,php,ruby}]
                        [-t {attached,combined,exported}]
                        [--skip-warnings] [-h]

porter save <estimator> [-l {c,go,java,js,php,ruby}]
                        [-t {attached,combined,exported}]
                        [--directory DIRECTORY]
                        [--skip-warnings] [-h]
```

First save the trained estimator to a file (see [model persistence](https://scikit-learn.org/stable/model_persistence.html)):

```python
from joblib import dump

dump(clf, 'estimator.joblib', compress=0)
```

Then port the saved estimator with the subcommand `port`:

```bash
porter port estimator.joblib -l js -t attached > estimator.js
```

The output can be passed to other tools, e.g. to [UglifyJS](https://github.com/mishoo/UglifyJS) to minify the code:

```bash
porter port estimator.joblib -l js -t attached | uglifyjs --compress -o estimator.min.js
```


## Known Issues

- The ported support vector machines, [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html), can make different predictions than the original estimators in rare cases. Therefore sklearn-porter raises a `QualityWarning` for them: check the ported estimator with the `test` method.
- The [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) is ported with the `SAMME` algorithm, because scikit-learn removed the `SAMME.R` algorithm.


## Development

### Setup

The Python version and all dependencies are managed with [uv](https://docs.astral.sh/uv/). Install everything with:

```bash
uv sync --all-groups
```

### Commands

```bash
make lint             # Check the code and its format with ruff
make format           # Fix and format the code with ruff
make test             # Run the fast tests
make test-regression  # Run the regression tests in Docker
make test-all         # Run all tests with coverage in Docker
make notebook         # Start Jupyter with the examples
make examples         # Generate the example notebooks with jupytext
```

### Testing

The fast tests check the Python code without running the ported estimators. GitHub Actions runs them on every push and pull request with Python 3.11 to 3.14 ([workflow](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml)):

```bash
uv run pytest
```

The regression tests compile and run the ported estimators and compare their predictions with the original estimators. They take a long time (10+ minutes) and need all compilers and interpreters. Therefore they are marked with `regression`, skipped by default and only run locally in [Docker](Dockerfile):

```bash
make test-regression
```

The tests use the Python version from [`.python-version`](.python-version). Set `PYTHON_VERSION` to use another one:

```bash
make test-regression PYTHON_VERSION=3.14
```

The Docker image (Debian 12 "bookworm") contains these compilers and interpreters:

| Language   | Compiler / Interpreter |
|------------|------------------------|
| C          | GCC 12                 |
| Go         | Go 1.19                |
| Java       | OpenJDK 17             |
| JavaScript | Node.js 18             |
| PHP        | PHP 8.2                |
| Ruby       | Ruby 3.1               |

Please note that, in general, you can use older compilers and interpreters with the generated source code. For instance, you can use Java 8 to compile and run the models.

### Logging

Change the option `logging.level` to see what sklearn-porter does:

```python
from logging import DEBUG

from sklearn_porter import options

options['logging.level'] = DEBUG
```

### Contributing

Commit messages and pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/), e.g. `feat: add support for ...` or `fix(php): ...`. Pull requests are squash merged into `main`, so the pull request title becomes the commit message.

### Releases

Releases are created with [release-please](https://github.com/googleapis/release-please-action):

1. release-please collects the Conventional Commits on `main` in a release pull request. It updates the version and the [changelog](CHANGELOG.md).
2. Merging the release pull request creates the Git tag and the GitHub release.
3. The package is published to [TestPyPI](https://test.pypi.org/project/sklearn-porter/), installed and checked from there and only then published to [PyPI](https://pypi.org/project/sklearn-porter/).


## Citation

If you use sklearn-porter in your work, please cite it. You can use this BibTeX entry:

```bibtex
@misc{sklearn_porter,
  author = {Darius Morawiec},
  title = {sklearn-porter},
  note = {Transpile trained scikit-learn estimators to C, Go, Java, JavaScript, PHP and Ruby},
  url = {https://github.com/nok/sklearn-porter}
}
```


## License

sklearn-porter is open source software released under the [BSD 3-Clause](LICENSE) license.
