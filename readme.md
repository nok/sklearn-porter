
# sklearn-porter

[![Build Status stable branch](https://img.shields.io/travis/nok/sklearn-porter/stable.svg)](https://travis-ci.org/nok/sklearn-porter)
[![PyPI](https://img.shields.io/pypi/v/sklearn-porter.svg)](https://pypi.python.org/pypi/sklearn-porter)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators to C, Java, JavaScript and others.<br>It's recommended for limited embedded systems and critical applications where performance matters most.


## Machine learning algorithms

<table>
    <tbody>
        <tr>
            <td align="center" width="32%"><strong>Algorithm</strong></td>
            <td align="center" colspan="6" width="68%"><strong>Programming language</strong></td>
        </tr>
        <tr>
            <td align="left" width="32%">Classifier</td>
            <td align="center" width="13%">Java *</td>
            <td align="center" width="11%">JS</td>
            <td align="center" width="11%">C</td>
            <td align="center" width="11%">Go</td>
            <td align="center" width="11%">PHP</td>
            <td align="center" width="11%">Ruby</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">svm.SVC</a></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/java/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/SVC/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/js/basics.pct.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/c/basics.pct.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/php/basics.pct.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/ruby/basics.pct.ipynb">✓</a></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html">svm.NuSVC</a></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/java/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/NuSVC/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/js/basics.pct.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/c/basics.pct.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/php/basics.pct.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/ruby/basics.pct.ipynb">✓</a></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">svm.LinearSVC</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/java/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/LinearSVC/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/js/basics.pct.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/c/basics.pct.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/go/basics.pct.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/php/basics.pct.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/ruby/basics.pct.ipynb">✓</a></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">tree.DecisionTreeClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/java/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/DecisionTreeClassifier/java/basics_embedded.pct.ipynb">✓ ᴱ</a>, <a href="examples/estimator/classifier/DecisionTreeClassifier/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/js/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/DecisionTreeClassifier/js/basics_embedded.pct.ipynb">✓ ᴱ</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/c/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/DecisionTreeClassifier/c/basics_embedded.pct.ipynb">✓ ᴱ</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/go/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/DecisionTreeClassifier/go/basics_embedded.pct.ipynb">✓ ᴱ</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/php/basics.pct.ipynb">✓</a>,  <a href="examples/estimator/classifier/DecisionTreeClassifier/php/basics_embedded.pct.ipynb">✓ ᴱ</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/ruby/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/DecisionTreeClassifier/ruby/basics_embedded.pct.ipynb">✓ ᴱ</a></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">ensemble.RandomForestClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/RandomForestClassifier/java/basics_embedded.pct.ipynb">✓ ᴱ</a>, <a href="examples/estimator/classifier/RandomForestClassifier/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/RandomForestClassifier/js/basics_embedded.pct.ipynb">✓ ᴱ</a></td>
            <td align="center"><a href="examples/estimator/classifier/RandomForestClassifier/c/basics_embedded.pct.ipynb">✓ ᴱ</a></td>
            <td align="center">✓ ᴱ</td>
            <td align="center">✓ ᴱ</td>
            <td align="center">✓ ᴱ</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">ensemble.ExtraTreesClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/ExtraTreesClassifier/java/basics_embedded.pct.ipynb">✓ ᴱ</a>, <a href="examples/estimator/classifier/ExtraTreesClassifier/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/ExtraTreesClassifier/js/basics.pct.ipynb">✓ ᴱ</a></td>
            <td align="center"><a href="examples/estimator/classifier/ExtraTreesClassifier/c/basics.pct.ipynb">✓ ᴱ</a></td>
            <td align="center"></td>
            <td align="center">✓ ᴱ</td>
            <td align="center">✓ ᴱ</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">ensemble.AdaBoostClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/AdaBoostClassifier/java/basics_embedded.pct.ipynb">✓ ᴱ</a>, <a href="examples/estimator/classifier/AdaBoostClassifier/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/AdaBoostClassifier/js/basics_embedded.pct.ipynb">✓ ᴱ</a>, <a href="examples/estimator/classifier/AdaBoostClassifier/js/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/AdaBoostClassifier/c/basics_embedded.pct.ipynb">✓ ᴱ</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">neighbors.KNeighborsClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/KNeighborsClassifier/java/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/KNeighborsClassifier/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/KNeighborsClassifier/js/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/KNeighborsClassifier/js/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB">naive_bayes.GaussianNB</a></td>
            <td align="center"><a href="examples/estimator/classifier/GaussianNB/java/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/GaussianNB/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/GaussianNB/js/basics.pct.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB">naive_bayes.BernoulliNB</a></td>
            <td align="center"><a href="examples/estimator/classifier/BernoulliNB/java/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/BernoulliNB/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/BernoulliNB/js/basics.pct.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">neural_network.MLPClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/MLPClassifier/java/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/MLPClassifier/java/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"><a href="examples/estimator/classifier/MLPClassifier/js/basics.pct.ipynb">✓</a>, <a href="examples/estimator/classifier/MLPClassifier/js/basics_imported.pct.ipynb">✓ ᴵ</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="left" width="32%">Regressor</td>
            <td colspan="6" width="68%"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html">neural_network.MLPRegressor</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/regressor/MLPRegressor/js/basics.pct.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>

✓ = is full-featured,　ᴱ = with embedded model data,　ᴵ = with imported model data,　* = default language


## Installation

### Stable

[![Build Status stable branch](https://img.shields.io/travis/nok/sklearn-porter/stable.svg)](https://travis-ci.org/nok/sklearn-porter)
[![PyPI](https://img.shields.io/pypi/v/sklearn-porter.svg)](https://pypi.python.org/pypi/sklearn-porter)
[![PyPI](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.python.org/pypi/sklearn-porter)

```bash
$ pip install sklearn-porter
```

### Development

[![Build Status master branch](https://img.shields.io/travis/nok/sklearn-porter/master.svg)](https://travis-ci.org/nok/sklearn-porter)

If you want the [latest changes](https://github.com/nok/sklearn-porter/blob/master/changelog.md#unreleased), you can install this package from the [master](https://github.com/nok/sklearn-porter/tree/master) branch:

```bash
$ pip uninstall -y sklearn-porter
$ pip install --no-cache-dir https://github.com/nok/sklearn-porter/zipball/master
```


## Usage


### Export

The following example demonstrates how you can transpile a [decision tree estimator](http://scikit-learn.org/stable/modules/tree.html#classification) to Java:

```python
from sklearn.datasets import load_iris
from sklearn.tree import tree
from sklearn_porter import Porter

# Load data and train the classifier:
samples = load_iris()
X, y = samples.data, samples.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Export:
porter = Porter(clf, language='java')
output = porter.export(embed_data=True)
print(output)
```

The exported [result](examples/estimator/classifier/DecisionTreeClassifier/java/basics_embedded.pct.py#L60-L110) matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the decision tree.


### Integrity

You should always check and compute the integrity between the original and the transpiled estimator:

```python
# ...
porter = Porter(clf, language='java')

# Compute integrity score:
integrity = porter.integrity_score(X)
print(integrity)  # 1.0
```


### Prediction

You can compute the prediction(s) in the target programming language:

```python
# ...
porter = Porter(clf, language='java')

# Prediction(s):
Y_java = porter.predict(X)
y_java = porter.predict(X[0])
y_java = porter.predict([1., 2., 3., 4.])
```

### Notebooks

You can run and test all notebooks by starting a Jupyter notebook server locally:

```bash
$ make open.examples
$ make stop.examples
```

## Command-line interface

In general you can use the porter on the command line:

```
$ porter <pickle_file> [--to <directory>]
         [--class_name <class_name>] [--method_name <method_name>]
         [--export] [--checksum] [--data] [--pipe]
         [--c] [--java] [--js] [--go] [--php] [--ruby]
         [--version] [--help]
```

The following example shows how you can save a trained estimator to the [pickle format](http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example):

```python
# ...

# Extract estimator:
joblib.dump(clf, 'estimator.pkl', compress=0)
```

After that the estimator can be transpiled to JavaScript by using the following command:

```bash
$ porter estimator.pkl --js
```

The target programming language is changeable on the fly:

```bash
$ porter estimator.pkl --c
$ porter estimator.pkl --java
$ porter estimator.pkl --php
$ porter estimator.pkl --java
$ porter estimator.pkl --ruby
```

For further processing the argument `--pipe` can be used to pass the result:

```bash
$ porter estimator.pkl --js --pipe > estimator.js
```

For instance the result can be minified by using [UglifyJS](https://github.com/mishoo/UglifyJS2):

```bash
$ porter estimator.pkl --js --pipe | uglifyjs --compress -o estimator.min.js
```


## Development


### Environment

You have to install required modules for broader development:

```bash
$ make install.environment  # conda environment (optional)
$ make install.requirements.development  # pip requirements
```

Independently, the following compilers and intepreters are required to cover all tests:

<table>
    <thead>
        <tr>
            <td width="33%"><strong>Name</strong></td>
            <td width="33%"><strong>Version</strong></td>
            <td width="33%"><strong>Command</strong></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://gcc.gnu.org">GCC</a></td>
            <td><code>>=4.2</code></td>
            <td><code>gcc --version</code></td>
        </tr>
        <tr>
            <td><a href="https://java.com">Java</a></td>
            <td><code>>=1.6</code></td>
            <td><code>java -version</code></td>
        </tr>
        <tr>
            <td><a href="http://www.php.net">PHP</a></td>
            <td><code>>=5.6</code></td>
            <td><code>php --version</code></td>
        </tr>
        <tr>
            <td><a href="https://www.ruby-lang.org">Ruby</a></td>
            <td><code>>=2.4.1</code></td>
            <td><code>ruby --version</code></td>
        </tr>
        <tr>
            <td><a href="https://golang.org">Go</a></td>
            <td><code>>=1.7.4</code></td>
            <td><code>go version</code></td>
        </tr>
        <tr>
            <td><a href="https://nodejs.org">Node.js</a></td>
            <td><code>>=6</code></td>
            <td><code>node --version</code></td>
        </tr>
    </tbody>
</table>


### Testing

The tests cover module functions as well as matching predictions of transpiled estimators. Start all tests with:

```bash
$ make test
```

The test files have a specific pattern: `'[Algorithm][Language]Test.py'`:

```bash
$ pytest tests -v -o python_files='RandomForest*Test.py'
$ pytest tests -v -o python_files='*JavaTest.py'
```

While you are developing new features or fixes, you can reduce the test duration by changing the number of tests:

```bash
$ N_RANDOM_FEATURE_SETS=5 N_EXISTING_FEATURE_SETS=10 \
  pytest tests -v -o python_files='*JavaTest.py'
```


### Quality

It's highly recommended to ensure the code quality. For that [Pylint](https://github.com/PyCQA/pylint/) is used. Start the linter with:

```bash
$ make lint
```


## Citation

[![GitHub license](https://img.shields.io/pypi/l/sklearn-porter.svg)](https://raw.githubusercontent.com/nok/sklearn-porter/master/license.txt)

If you use this implementation in you work, please add a reference/citation to the paper. You can use the following BibTeX entry:

```
@unpublished{skpodamo,
  author = {Darius Morawiec},
  title = {sklearn-porter},
  note = {Transpile trained scikit-learn estimators to C, Java, JavaScript and others},
  url = {https://github.com/nok/sklearn-porter}
}
```


## License

The module is Open Source Software released under the [MIT](license.txt) license.


## Questions?

[![Stack Overflow](https://img.shields.io/badge/Stack%20Overflow-Ask%20questions-blue.svg)](https://stackoverflow.com/questions/tagged/sklearn-porter)
[![Join the chat at https://gitter.im/nok/sklearn-porter](https://badges.gitter.im/nok/sklearn-porter.svg)](https://gitter.im/nok/sklearn-porter?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Twitter](https://img.shields.io/twitter/follow/darius_morawiec.svg?label=follow&style=popout)](https://twitter.com/darius_morawiec)

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/sklearn-porter).
