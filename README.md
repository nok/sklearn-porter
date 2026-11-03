<h1 align="center">sklearn-porter</h1>

<p align="center">
  Transpile trained <a href="https://github.com/scikit-learn/scikit-learn">scikit-learn</a> estimators and transformers<br>
  to C, C#, Go, Java, JavaScript, PHP and Ruby.
</p>

<p align="center">
  <a href="https://github.com/nok/sklearn-porter/actions/workflows/ci.yml"><img src="https://github.com/nok/sklearn-porter/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/sklearn-porter/"><img src="https://img.shields.io/pypi/v/sklearn-porter.svg?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/sklearn-porter/"><img src="https://img.shields.io/pypi/pyversions/sklearn-porter.svg" alt="Python versions"></a>
  <a href="https://pypistats.org/packages/sklearn-porter"><img src="https://img.shields.io/pypi/dm/sklearn-porter.svg?color=blue" alt="Downloads"></a>
  <a href="https://github.com/nok/sklearn-porter/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/sklearn-porter.svg?color=blue" alt="License"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#examples">Examples</a> •
  <a href="#support">Support</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
Use the generated code to make predictions where Python isn't available or is too heavy, e.g. on embedded systems, in web browsers or in existing applications. The ported estimators make the same predictions as the original scikit-learn estimators, the ported transformers (e.g. for the preprocessing) transform the features like the original transformers.
</p>


## Installation

```bash
uv add sklearn-porter
```

```bash
pip install sklearn-porter
```

Python 3.11 or newer and scikit-learn 1.8 or newer are required.


## Quick Start

<!-- fmt:off -->
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn_porter import make, port, save, test

# Train a classifier:
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)

# Port it to JavaScript and get the source code:
code = port(clf, language='js', template='combined')

# Save it to JavaScript files (the code and the model data):
src_path, json_path = save(clf, language='js', template='exported', directory='/tmp')

# Make predictions with the ported classifier (with Node.js):
y_pred, y_proba = make(clf, X[:3], language='js')
print(y_pred)  # [0 0 0]

# Check that all predictions match those of scikit-learn:
print(test(clf, X, language='js'))  # 1.0
```
<!-- fmt:on -->

<details>
<summary>The generated JavaScript code (excerpt)</summary>

```js
this._compute = function(features, post) {
	var classes = new Array(3).fill(0);
	if (features[3] <= 0.8000000417232512) {
		classes[0] = 1.0;
	} else {
		if (features[3] <= 1.7500000596046448 || isNaN(features[3])) {
			classes[1] = 0.9074074074074074;
			classes[2] = 0.09259259259259259;
		} else {
			classes[1] = 0.021739130434782608;
			classes[2] = 0.9782608695652174;
		}
	}
	return post(classes);
};
```

Run it with the features of a sample:

```console
$ node DecisionTreeClassifier.js 6.3 3.3 6.0 2.5
Predicted class: 2
Probability of class #0 : 0
Probability of class #1 : 0.021739130434782608
Probability of class #2 : 0.9782608695652174
```
</details>

`make()` and `test()` need the compiler or interpreter of the language (see [languages](#languages)).


## Examples

- [Getting Started](https://github.com/nok/sklearn-porter/blob/main/examples/getting_started.ipynb) ✨: port a classifier, make predictions and check them with the functional and the object-oriented API.
- [Transformers](https://github.com/nok/sklearn-porter/blob/main/examples/transformers.ipynb): port transformers (e.g. a `StandardScaler` or a `ColumnTransformer`) and use the ported transformers.
- [Pipelines](https://github.com/nok/sklearn-porter/blob/main/examples/pipelines.ipynb): port a `Pipeline` of transformers and a final estimator (also of a `GridSearchCV`) to one source file.
- [CLI](https://github.com/nok/sklearn-porter/blob/main/examples/cli.ipynb): port a saved classifier with the command `porter`, save the source code to files and run it.


## Support

### Estimators

<details open>
<summary><strong>Classifiers</strong> (21)</summary>

- [ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [ensemble.ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
- [ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [ensemble.HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [linear_model.LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
- [linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- [linear_model.RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
- [linear_model.RidgeClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html)
- [linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) ²
- [naive_bayes.BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) ¹
- [naive_bayes.ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html) ¹
- [naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) ¹
- [naive_bayes.MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) ¹
- [neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) ¹
- [neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) ¹
- [svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) ¹
- [svm.NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) ¹ ²
- [svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) ¹ ²
- [tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
</details>

<details>
<summary><strong>Regressors</strong> (21)</summary>

- [ensemble.ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [ensemble.HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [linear_model.ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)
- [linear_model.BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
- [linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [linear_model.ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
- [linear_model.HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
- [linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [linear_model.LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
- [linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [linear_model.RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
- [linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
- [neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) ¹
- [neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) ¹
- [svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)
- [svm.NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html) ¹
- [svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) ¹
- [tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
</details>

<details>
<summary><strong>Outlier detectors</strong> (2)</summary>

- [ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [svm.OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) ¹
</details>

### Transformers

<details>
<summary><strong>Composition</strong> (2)</summary>

- [compose.ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [pipeline.FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)
</details>

<details>
<summary><strong>Feature extraction</strong> (4)</summary>

- [feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [feature_extraction.text.HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)
- [feature_extraction.text.TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
- [feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
</details>

<details>
<summary><strong>Feature selection</strong> (11)</summary>

- [feature_selection.GenericUnivariateSelect](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html)
- [feature_selection.RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
- [feature_selection.RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
- [feature_selection.SelectFdr](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html)
- [feature_selection.SelectFpr](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html)
- [feature_selection.SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
- [feature_selection.SelectFwe](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html)
- [feature_selection.SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
- [feature_selection.SelectPercentile](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html)
- [feature_selection.SequentialFeatureSelector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)
- [feature_selection.VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)
</details>

<details>
<summary><strong>Preprocessing</strong> (15)</summary>

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
</details>

¹ without the template `combined`, ² without `predict_proba` (even if scikit-learn provides it, e.g. `SVC(probability=True)`). All other estimators support all languages and templates, the classifiers also `predict_proba` (if scikit-learn provides it). `porter show` lists all supported combinations.

### Languages

| Language   | Key    | `make()` and `test()` need | Tested with           |
|------------|--------|----------------------------|-----------------------|
| C          | `c`    | `gcc`                      | GCC 14                |
| C#         | `cs`   | `dotnet` (.NET SDK 8+)     | .NET SDK 10           |
| Go         | `go`   | `go`                       | Go 1.27               |
| Java       | `java` | `javac` and `java`         | OpenJDK 21, Gson 2.15 |
| JavaScript | `js`   | `node`                     | Node.js 24            |
| PHP        | `php`  | `php`                      | PHP 8.4               |
| Ruby       | `ruby` | `ruby`                     | Ruby 3.3              |

The key selects the language, e.g. `port(clf, language='cs')`.

### Templates

The template defines how the model data is stored in the generated code:

- `attached`: The model data is embedded in the source code, e.g. as arrays.
- `exported`: The model data is saved in a separate JSON file, which the source code reads at runtime.
- `combined`: The model data is compiled into the logic of the source code, e.g. as nested if-else statements.

The template is selected like the language, e.g. `port(clf, template='exported')`.


## Documentation

Read these pages before you use a ported estimator in production:

- [Estimator notes](https://github.com/nok/sklearn-porter/blob/main/docs/estimator-notes.md): How each estimator and transformer is ported and which parameters can lead to different predictions.
- [Limitations](https://github.com/nok/sklearn-porter/blob/main/docs/limitations.md): The known limitations, e.g. of pipelines, large Java methods, C#, sparse data and Windows.
- [Changelog](https://github.com/nok/sklearn-porter/blob/main/CHANGELOG.md): All changes of each version.


## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/nok/sklearn-porter/blob/main/CONTRIBUTING.md) for the development setup, the tests and the release process.


## Citation

If you use sklearn-porter in your work, please cite it. GitHub shows the citation of the [citation file](https://github.com/nok/sklearn-porter/blob/main/CITATION.cff) (with the version) under "Cite this repository", or use this BibTeX entry:

```bibtex
@software{sklearn_porter,
  author = {Darius Morawiec},
  title = {sklearn-porter},
  note = {Transpile trained scikit-learn estimators and transformers to C, C#, Go, Java, JavaScript, PHP and Ruby},
  url = {https://github.com/nok/sklearn-porter}
}
```


## License

sklearn-porter is open source software released under the [BSD 3-Clause](https://github.com/nok/sklearn-porter/blob/main/LICENSE) license.
