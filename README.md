# sklearn-porter

[![CI](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml/badge.svg)](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/sklearn-porter.svg?color=blue)](https://pypi.org/project/sklearn-porter/)
[![Python versions](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.org/project/sklearn-porter/)
[![License](https://img.shields.io/pypi/l/sklearn-porter.svg?color=blue)](https://github.com/nok/sklearn-porter/blob/main/LICENSE)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators to **C**, **Go**, **Java**, **JavaScript**, **PHP** and **Ruby**.

Use the generated code to make predictions where Python isn't available or is too heavy, e.g. on embedded systems, in web browsers or in existing applications. The ported estimators make the same predictions as the original scikit-learn estimators.

Navigation: [Installation](#installation) • [Support](#support) • [Usage](#usage) • [CLI](#cli) • [Documentation](#documentation) • [Contributing](#contributing) • [Citation](#citation) • [License](#license)


## Installation

```bash
uv add sklearn-porter
```

```bash
pip install sklearn-porter
```

Python 3.11 or newer and scikit-learn 1.8 or newer are required.


## Support

### Estimators

#### Classifiers

- [svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [svm.NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)
- [svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [linear_model.RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
- [linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- [tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [ensemble.ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
- [ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [naive_bayes.BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
- [naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

#### Regressors

- [tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [ensemble.ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

### Languages

The estimators are ported to **C**, **Go**, **Java**, **JavaScript**, **PHP** and **Ruby**.

### Templates

The template defines how the model data is stored in the generated code:

- `attached`: the model data is embedded in the code.
- `exported`: the model data is saved in a separate JSON file.
- `combined`: the model data is compiled into the logic of the code (e.g. nested if-else statements).


## Usage

<!-- fmt:off -->
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
src_path, json_path = save(clf, language='js', template='exported', directory='/tmp')
print(src_path, json_path)

# 4. Make predictions with the ported classifier:
y_classes, y_probas = make(clf, X[:10], language='js', template='exported')
print(y_classes, y_probas)

# 5. Check the ported classifier: the score is the fraction of predictions
#    that match those of the original classifier.
score = test(clf, X[:10], language='js', template='exported')
print(score)
```
<!-- fmt:on -->

Steps 4 and 5 require a compiler or interpreter for the target language (e.g. `node` for JavaScript).

The [example notebook](https://github.com/nok/sklearn-porter/blob/main/examples/basics/index.pct.ipynb) shows the functional and the object-oriented API step by step.


## CLI

After installation, the `porter` command is available:

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

First, save the trained estimator to a file (see [model persistence](https://scikit-learn.org/stable/model_persistence.html)):

```python
from joblib import dump

dump(clf, 'estimator.joblib')
```

Then port the saved estimator with the `port` subcommand:

```bash
porter port estimator.joblib -l js -t attached > estimator.js
```

The output can be passed to other tools, e.g. to [terser](https://github.com/terser/terser), to minify the code:

```bash
porter port estimator.joblib -l js -t attached | npx terser --compress -o estimator.min.js
```


## Documentation

Read these pages before you use a ported estimator in production:

- [Estimator Notes](https://github.com/nok/sklearn-porter/blob/main/docs/estimator-notes.md): multilabel and multi-output classification and the parameters of single estimators which can't be ported or which can lead to different predictions.
- [Limitations](https://github.com/nok/sklearn-porter/blob/main/docs/limitations.md): pipelines, the size of Java methods and sparse data.
- [Known Issues](https://github.com/nok/sklearn-porter/blob/main/docs/known-issues.md): the ported classifiers return class indices instead of labels.


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
