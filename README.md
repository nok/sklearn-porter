# sklearn-porter

[![CI](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml/badge.svg)](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/sklearn-porter.svg?color=blue)](https://pypi.org/project/sklearn-porter/)
[![Python versions](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.org/project/sklearn-porter/)
[![License](https://img.shields.io/pypi/l/sklearn-porter.svg?color=blue)](https://github.com/nok/sklearn-porter/blob/main/LICENSE)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators to C, Go, Java, JavaScript, PHP and Ruby.

Use the generated code to make predictions where Python isn't available or too heavy, e.g. on embedded systems, in web browsers or in existing applications. The ported estimators make the same predictions as the original estimators of scikit-learn.

Navigation: [Estimators](#estimators) • [Installation](#installation) • [Usage](#usage) • [Known Issues](#known-issues) • [Contributing](#contributing) • [Citation](#citation) • [License](#license)


## Estimators

### Classifiers

- [svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [svm.NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)
- [svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [ensemble.ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
- [ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [naive_bayes.BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
- [naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

### Regressors

- [tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [ensemble.ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

### Languages, templates and methods

The estimators are ported to **C**, **Go**, **Java**, **JavaScript**, **PHP** and **Ruby**. The template defines how the model data is stored in the generated code:

- `attached`: the model data is part of the code.
- `exported`: the model data is saved in a separate JSON file.
- `combined`: the model data is compiled into the logic of the code (e.g. nested if-else statements).

All estimators support the method `predict`, most classifiers also `predict_proba`. Not every estimator supports every combination of language, template and method yet. Check a combination with [`Estimator.can`](#supported-combinations) or show all supported combinations with [`show`](#supported-combinations) or `porter show`.


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

### Regressors

Regressors are ported the same way. `make` returns the predicted values and `None` instead of probabilities. `test` returns the share of predictions which are close to the predictions of the original regressor (see [`numpy.isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)), because the floating point arithmetic of the target languages can differ slightly:

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from sklearn_porter import Estimator

X, y = load_diabetes(return_X_y=True)
reg = RandomForestRegressor(n_estimators=10, max_depth=5)
reg.fit(X, y)

est = Estimator(reg, language='java', template='attached')
y_values, _ = est.make(X[:10])
print(y_values)
print(est.test(X[:10]))
```

### Options

`port`, `save`, `make`, `test` and `Estimator` accept these additional arguments:

- `class_name`: the name of the class (or function) in the generated code. By default it's the name of the estimator, e.g. `DecisionTreeClassifier`.
- `converter`: a function which converts each floating point number of the model data to a string. By default it's `str`. Fewer digits make the code smaller, but can change the predictions, so check the result with `test`:

```python
output = port(
    clf,
    language='c',
    template='attached',
    class_name='Iris',
    converter=lambda value: f'{value:.6f}',
)
```

### Supported combinations

`show` prints the supported estimators, languages and templates, `Estimator.can` checks a single combination:

```python
from sklearn_porter import Estimator, show

print(show())  # or e.g. show(language='js')

est = Estimator(clf)
est.can(language='js', template='combined')  # True
est.can(language='js', template='combined', method='predict_proba')  # True
```

### Pipelines and optimizers

You can pass a fitted [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html). sklearn-porter extracts the final estimator of the pipeline or the best estimator of the optimizer and ports only this estimator (see [Known Issues](#known-issues)).

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

The output can be passed to other tools, e.g. to [terser](https://github.com/terser/terser) to minify the code:

```bash
porter port estimator.joblib -l js -t attached | npx terser --compress -o estimator.min.js
```

### Logging

Change the option `logging.level` to see what sklearn-porter does:

```python
from logging import DEBUG

from sklearn_porter import options

options['logging.level'] = DEBUG
```

### Examples

The [example notebook](https://github.com/nok/sklearn-porter/blob/main/examples/basics/index.pct.ipynb) shows the functional and the object-oriented API step by step.


## Known Issues

- Only the final estimator of a [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) is ported. Preprocessing steps like a [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) are dropped without a warning, so you have to apply them to the input data of the ported estimator yourself. Otherwise the predictions differ from the predictions of the pipeline.
- The ported support vector machines, [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html), can make different predictions than the original estimators in rare cases. Therefore sklearn-porter raises a `QualityWarning` for them: check the ported estimator with the `test` method.
- The [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) is ported with the `SAMME` algorithm, because scikit-learn removed the `SAMME.R` algorithm.


## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/nok/sklearn-porter/blob/main/CONTRIBUTING.md) for the development setup, the tests and the release process.


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

sklearn-porter is open source software released under the [BSD 3-Clause](https://github.com/nok/sklearn-porter/blob/main/LICENSE) license.
