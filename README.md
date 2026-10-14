# sklearn-porter

[![CI](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml/badge.svg)](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/sklearn-porter.svg?color=blue)](https://pypi.org/project/sklearn-porter/)
[![Python versions](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.org/project/sklearn-porter/)
[![License](https://img.shields.io/pypi/l/sklearn-porter.svg?color=blue)](https://github.com/nok/sklearn-porter/blob/main/LICENSE)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators to **C**, **Go**, **Java**, **JavaScript**, **PHP** and **Ruby**.

Use the generated code to make predictions where Python isn't available or is too heavy, e.g. on embedded systems, in web browsers or in existing applications. The ported estimators make the same predictions as the original scikit-learn estimators.

Navigation: [Installation](#installation) • [Support](#support) • [Usage](#usage) • [CLI](#cli) • [Limitations](#limitations) • [Known Issues](#known-issues) • [Contributing](#contributing) • [Citation](#citation) • [License](#license)


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

The classifiers `RidgeClassifier` and `MLPClassifier` also support multilabel classification, i.e. if they were fitted on a label indicator matrix, the ported `predict` returns one binary value per label.

The classifiers `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier` and `KNeighborsClassifier` also support multi-output classification, i.e. if they were fitted on a target with several columns, the ported `predict` returns one class per output. Like in scikit-learn, the ported `predict_proba` returns the class probabilities of each output.

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

# 5. Check the ported classifier: the score is the fraction of predictions
#    that match those of the original classifier.
score = test(clf, X[:10], language='js', template='exported')
print(score)
```

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


## Limitations

- Only the final estimator of a [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) is ported. Preprocessing steps like a [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) are silently dropped, so you have to apply them to the input data of the ported estimator yourself. Otherwise, the predictions differ from those of the pipeline.
- Java limits the bytecode of a method to 64 KB. Large estimators ported to Java with the `attached` or `combined` template can therefore fail to compile with a `code too large` error. Use the `exported` template instead.
- Sparse matrices are always converted to dense arrays, because the ported estimators only support dense data. This applies to estimators fitted on sparse data (e.g. [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) or [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)), to estimators converted with `sparsify()` (e.g. [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)) and to the samples passed to `make()` and `test()`. With many features (e.g. text features of a `TfidfVectorizer`) the dense model data, the ported source code and the exported JSON file can get very large, so make sure that enough memory (RAM) and disk space are available, both for porting and for compiling and running the ported estimator.


## Known Issues

- The ported classifiers return the index of the predicted class instead of its label. The predictions only match those of scikit-learn if the labels are `0, 1, …, n_classes - 1`, so encode other labels (e.g. `1..3`, `-1/+1` or strings) with a [`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) before training.
- [BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) with feature or class probabilities of 0 or 1 (e.g. with `alpha=0`) can't be ported and raises a `NotSupportedYetError`, because scikit-learn itself computes undefined (`NaN`) joint log likelihoods with them.
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) with `break_ties=True` can't be ported and raise a `NotSupportedYetError`.
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) can only be ported with `weights='uniform'` and the metrics `minkowski` (e.g. `euclidean` or `manhattan`), `chebyshev`, `cosine` and `hamming`. If several neighbors have the same distance to a sample (which is common with `hamming`), the ported estimator picks the first ones in the order of the training data, scikit-learn may pick others and predict a different class. Other metrics (e.g. `canberra` or a callable) and the weighted Minkowski distance (`metric_params={'w': ...}`) raise a `NotSupportedYetError`.


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
