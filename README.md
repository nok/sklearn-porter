# sklearn-porter

[![CI](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml/badge.svg)](https://github.com/nok/sklearn-porter/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/sklearn-porter.svg?color=blue)](https://pypi.org/project/sklearn-porter/)
[![Python versions](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.org/project/sklearn-porter/)
[![License](https://img.shields.io/pypi/l/sklearn-porter.svg?color=blue)](https://github.com/nok/sklearn-porter/blob/main/LICENSE)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators and transformers to **C**, **C#**, **Go**, **Java**, **JavaScript**, **PHP** and **Ruby**.

Use the generated code to make predictions where Python isn't available or is too heavy, e.g. on embedded systems, in web browsers or in existing applications. The ported estimators make the same predictions as the original scikit-learn estimators, the ported transformers (e.g. for the preprocessing) transform the features like the original transformers.

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
- [linear_model.LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
- [linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [linear_model.RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
- [linear_model.RidgeClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html)
- [linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- [tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [ensemble.ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
- [ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [ensemble.HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [naive_bayes.BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
- [naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [naive_bayes.MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [naive_bayes.ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
- [neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

#### Regressors

- [tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [ensemble.ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [ensemble.HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [linear_model.RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
- [linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [linear_model.LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
- [linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [linear_model.ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
- [linear_model.BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
- [linear_model.ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)
- [linear_model.HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
- [linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
- [svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)
- [svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [svm.NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html)
- [neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

#### Outlier detectors

- [ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [svm.OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)

### Transformers

#### Composition

- [compose.ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [pipeline.FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)

#### Preprocessing

- [impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [preprocessing.Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)
- [preprocessing.FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
- [preprocessing.KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
- [preprocessing.MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
- [preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- [preprocessing.Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)
- [preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
- [preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [preprocessing.PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
- [preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
- [preprocessing.RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
- [preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [preprocessing.TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)

### Languages

The estimators and transformers are ported to **C** (`c`), **C#** (`cs`), **Go** (`go`), **Java** (`java`), **JavaScript** (`js`), **PHP** (`php`) and **Ruby** (`ruby`). The keys in parentheses select the language, e.g. `port(clf, language='cs')`.

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

Steps 4 and 5 require a compiler or interpreter for the target language (e.g. `node` for JavaScript or the .NET SDK for C#).

The [example notebook](https://github.com/nok/sklearn-porter/blob/main/examples/basics/index.pct.ipynb) shows the functional and the object-oriented API step by step.

### Transformers

A ported transformer has the method `transform`, which returns the transformed features of a sample, analogous to `predict` of a ported estimator. The functions `port`, `save`, `make` and `test` accept transformers too:

<!-- fmt:off -->
```python
from sklearn.preprocessing import StandardScaler
from sklearn_porter import Transformer

# 1. Train a transformer:
scaler = StandardScaler()
scaler.fit(X)

# 2. Port the transformer and save it:
transformer = Transformer(scaler, language='js')
print(transformer.port())
transformer.save(directory='/tmp')  # /tmp/StandardScaler.js

# 3. Transform the features with the ported transformer and check them:
X_tr = transformer.transform(X[:10])
score = transformer.test(X[:10])
```
<!-- fmt:on -->

The ported transformer prints the transformed features of each sample in one line (separated by spaces), which is also the input format of the ported estimators.

### Pipelines

A [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) of supported transformers and a supported final estimator (also as the best estimator of a `GridSearchCV` or `RandomizedSearchCV`) is ported to one source file: the ported final estimator, which transforms the features of each sample with the ported transformers before its prediction. The transformers can change the number of features (e.g. a `OneHotEncoder`), the ported pipeline reads the features of the pipeline. The functions `port`, `save`, `make` and `test` accept pipelines too:

<!-- fmt:off -->
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn_porter import make, save, test

pipe = make_pipeline(StandardScaler(), SVC())
pipe.fit(X, y)

path = save(pipe, language='js', directory='/tmp')  # /tmp/Pipeline.js
y_classes, _ = make(pipe, X[:10], language='js')  # like pipe.predict(X[:10])
score = test(pipe, X[:10], language='js')
```
<!-- fmt:on -->

The generated file is used like a ported estimator:

```bash
node /tmp/Pipeline.js < samples.txt
```

### Object-oriented API

The central class `Porter` accepts an estimator, a transformer or a pipeline and creates an `Estimator`, a `Transformer` or a `Pipeline`:

<!-- fmt:off -->
```python
from sklearn_porter import Porter

porter = Porter(clf, language='java')  # creates an `Estimator`
y_classes, y_probas = porter.predict(X[:10])

porter = Porter(scaler, language='java')  # creates a `Transformer`
X_tr = porter.transform(X[:10])

porter = Porter(pipe, language='java')  # creates a `Pipeline`
y_classes, y_probas = porter.predict(X[:10])
```
<!-- fmt:on -->


## CLI

After installation, the `porter` command is available:

```
porter {show,port,save} [-h] [-v]

porter show [-l {c,cs,go,java,js,php,ruby}] [-h]

porter port <estimator> [-l {c,cs,go,java,js,php,ruby}]
                        [-t {attached,combined,exported}]
                        [--ignore-preprocessing]
                        [--skip-warnings] [-h]

porter save <estimator> [-l {c,cs,go,java,js,php,ruby}]
                        [-t {attached,combined,exported}]
                        [--directory DIRECTORY]
                        [--ignore-preprocessing]
                        [--skip-warnings] [-h]
```

First, save the trained estimator to a file (see [model persistence](https://scikit-learn.org/stable/model_persistence.html)):

```python
from joblib import dump

dump(clf, 'estimator.joblib')
```

Then port the saved estimator (or transformer or pipeline) with the `port` subcommand:

```bash
porter port estimator.joblib -l js -t attached > estimator.js
```

The output can be passed to other tools, e.g. to [terser](https://github.com/terser/terser), to minify the code:

```bash
porter port estimator.joblib -l js -t attached | npx terser --compress -o estimator.min.js
```


## Documentation

Read these pages before you use a ported estimator in production:

- [Estimator Notes](https://github.com/nok/sklearn-porter/blob/main/docs/estimator-notes.md): the labels of the classes, multilabel and multi-output classification and the parameters of single estimators and transformers which can't be ported or which can lead to different predictions.
- [Limitations](https://github.com/nok/sklearn-porter/blob/main/docs/limitations.md): pipelines, the types of the labels, the size of Java methods, the requirements of C#, sparse data and Windows.


## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/nok/sklearn-porter/blob/main/CONTRIBUTING.md) for the development setup, the tests and the release process.


## Citation

If you use sklearn-porter in your work, please cite it. You can use this BibTeX entry:

```bibtex
@misc{sklearn_porter,
  author = {Darius Morawiec},
  title = {sklearn-porter},
  note = {Transpile trained scikit-learn estimators to C, C#, Go, Java, JavaScript, PHP and Ruby},
  url = {https://github.com/nok/sklearn-porter}
}
```


## License

sklearn-porter is open source software released under the [BSD 3-Clause](https://github.com/nok/sklearn-porter/blob/main/LICENSE) license.
