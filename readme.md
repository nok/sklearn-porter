
# sklearn-porter

[![CI](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml/badge.svg)](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sklearn-porter.svg?color=blue)](https://pypi.python.org/pypi/sklearn-porter)
[![PyPI](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.python.org/pypi/sklearn-porter)
[![GitHub license](https://img.shields.io/pypi/l/sklearn-porter.svg?color=blue)](https://raw.githubusercontent.com/nok/sklearn-porter/main/LICENSE)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators to C, Java, JavaScript and others.<br>It's recommended for limited embedded systems and critical applications where performance matters most.

Navigation: [Estimators](#estimators) • [Installation](#installation) • [Usage](#usage) • [Known Issues](#known-issues) • [Development](#development) • [Citation](#citation) • [License](#license)


## Estimators

This table gives an overview over all supported combinations of estimators, programming languages and templates.

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
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">svm.SVC</a>
    </td>
    <td>✓</td>
    <td></td>
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
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html">svm.NuSVC</a>
    </td>
    <td>✓</td>
    <td></td>
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
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">svm.LinearSVC</a>
    </td>
    <td>✓</td>
    <td></td>
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
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">tree.DecisionTreeClassifier</a>
    </td>
    <td>✓ᴾ</td>
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
    <td>✓ᴾ</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">ensemble.RandomForestClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">ensemble.ExtraTreesClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">ensemble.AdaBoostClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">neighbors.KNeighborsClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
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
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB">naive_bayes.BernoulliNB</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB">naive_bayes.GaussianNB</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">neural_network.MLPClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html">neural_network.MLPRegressor</a>
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

✓ = support of `predict`,　ᴾ = support of `predict_proba`,　× = not supported or feasible<br>
ᴀ = attached model data,　ᴇ = exported model data (JSON),　ᴄ = combined model data


## Installation

```bash
# with uv:
uv add sklearn-porter

# or with pip:
pip install sklearn-porter
```

The prerequisites are Python `>= 3.11` and `scikit-learn >= 1.8`. For older versions of scikit-learn you can use [sklearn-porter v0.7.4](https://pypi.org/project/sklearn-porter/0.7.4/).

To compile and run the ported estimators with `make` and `test` you need the compiler or interpreter of the target language, e.g. `node` for JavaScript.


## Usage

### Basics

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter import port, save, make, test

# 1. Load data and train a dummy classifier:
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)

# 2. Port or transpile an estimator:
output = port(clf, language='js', template='attached')
print(output)

# 3. Save the ported estimator:
src_path, json_path = save(
    clf, language='js', template='exported', directory='/tmp'
)
print(src_path, json_path)

# 4. Make predictions with the ported estimator:
y_classes, y_probas = make(clf, X[:10], language='js', template='exported')
print(y_classes, y_probas)

# 5. Test always the ported estimator by making an integrity check:
score = test(clf, X[:10], language='js', template='exported')
print(score)
```

### OOP

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter import Estimator

# 1. Load data and train a dummy classifier:
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)

# 2. Port or transpile an estimator:
est = Estimator(clf, language='js', template='attached')
output = est.port()
print(output)

# 3. Save the ported estimator:
est.template = 'exported'
src_path, json_path = est.save(directory='/tmp')
print(src_path, json_path)

# 4. Make predictions with the ported estimator:
y_classes, y_probas = est.make(X[:10])
print(y_classes, y_probas)

# 5. Test always the ported estimator by making an integrity check:
score = est.test(X[:10])
print(score)
```

### CLI

In addition you can use the sklearn-porter on the command line. The command calls `porter` and is available after the installation.

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

You can serialize an estimator and save it locally. For more details you can read the instructions to  [model persistence](http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example).

```python
from joblib import dump

dump(clf, 'estimator.joblib', compress=0)
```

After that the estimator can be transpiled by using the subcommand `port`:

```bash
porter port estimator.joblib -l js -t attached > estimator.js
```

For further processing you can pass the result to another applications, e.g. [UglifyJS](https://github.com/mishoo/UglifyJS2).

```bash
porter port estimator.joblib -l js -t attached | uglifyjs --compress -o estimator.min.js
```

## Known Issues

- In some rare cases the regression tests of the support vector machines, [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html), fail. Because of that a `QualityWarning` will be raised which should remind you to evaluate the result by using the `test` method.
- The [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) is ported with the `SAMME` algorithm, because scikit-learn removed `SAMME.R`.


## Development

### Setup

The Python version and all dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

### Commands

```bash
make lint             # Lint and check the format with ruff
make format           # Fix and format the code with ruff
make test             # Run the fast tests on the host
make test-regression  # Run the regression tests in Docker
make test-all         # Run all tests with coverage in Docker
make notebook         # Start a Jupyter notebook with the examples
make examples         # Generate the notebooks with jupytext
```

### Testing

The fast tests run on every push and pull request with [GitHub Actions](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml) on Python 3.11 to 3.14:

```bash
uv run pytest
```

The regression tests compile and execute the ported estimators and compare their predictions with the original estimators of scikit-learn. They take a long time and require all compilers and interpreters, so they are marked with `regression`, skipped by default and only run locally in a [Docker](Dockerfile) container:

```bash
make test-regression
```

The Docker image contains these compilers and interpreters (Debian bookworm):

| Language   | Compiler / Interpreter |
|------------|------------------------|
| C          | GCC 12                 |
| Go         | Go 1.19                |
| Java       | OpenJDK 17             |
| JavaScript | Node.js 18             |
| PHP        | PHP 8.2                |
| Ruby       | Ruby 3.1               |

In general you can use older compilers and interpreters with the generated source code.

### Logging

You can activate logging by changing the option `logging.level`.

```python
from sklearn_porter import options

from logging import DEBUG

options['logging.level'] = DEBUG
```

### Contributing

The commit messages and pull request titles follow the [Conventional Commits](https://www.conventionalcommits.org/) specification, e.g. `feat: add support for ...` or `fix(php): ...`. Pull requests are squash merged into `main`, so the pull request title becomes the commit message.

### Releases

Releases are created with [release-please](https://github.com/googleapis/release-please-action): it collects the Conventional Commits on `main` in a release pull request, which updates the version and the [changelog](CHANGELOG.md). Merging the release pull request creates the Git tag and the GitHub release. After that the package is published to [TestPyPI](https://test.pypi.org/project/sklearn-porter/), verified and finally published to [PyPI](https://pypi.org/project/sklearn-porter/).


## Citation

If you use this implementation in you work, please add a reference/citation to the paper. You can use the following BibTeX entry:

```bibtex
@unpublished{sklearn_porter,
  author = {Darius Morawiec},
  title = {sklearn-porter},
  note = {Transpile trained scikit-learn estimators to C, Java, JavaScript and others},
  url = {https://github.com/nok/sklearn-porter}
}
```


## License

The package is Open Source Software released under the [BSD 3-Clause](LICENSE) license.
