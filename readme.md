
# sklearn-porter

[![Build Status](https://img.shields.io/travis/nok/sklearn-porter/master.svg)](https://travis-ci.org/nok/sklearn-porter)
[![PyPI](https://img.shields.io/pypi/v/sklearn-porter.svg)](https://pypi.python.org/pypi/sklearn-porter)
[![PyPI](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.python.org/pypi/sklearn-porter)
[![GitHub license](https://img.shields.io/pypi/l/sklearn-porter.svg)](https://raw.githubusercontent.com/nok/sklearn-porter/master/license.txt)
[![Join the chat at https://gitter.im/nok/sklearn-porter](https://badges.gitter.im/nok/sklearn-porter.svg)](https://gitter.im/nok/sklearn-porter?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) models to [C](https://en.wikipedia.org/wiki/C_(programming_language)), [Java](https://en.wikipedia.org/wiki/Java_(programming_language)), [JavaScript](https://en.wikipedia.org/wiki/JavaScript) and others.<br>It's recommended for limited embedded systems and critical applications where performance matters most.


## Machine learning algorithms

<table>
    <tbody>
        <tr>
            <td align="center" width="40%"><strong>Algorithm</strong></td>
            <td align="center" colspan="6" width="60%"><strong>Programming language</strong></td>
        </tr>
        <tr>
            <td align="left" width="40%">Classification</td>
            <td align="center" width="10%">C</td>
            <td align="center" width="10%">Java</td>
            <td align="center" width="10%">JavaScript</td>
            <td align="center" width="10%">Go</td>
            <td align="center" width="10%">PHP</td>
            <td align="center" width="10%">Ruby</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.svm.SVC.html">sklearn.svm.SVC</a></td>
            <td align="center"><a href="examples/classifier/SVC/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/SVC/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/SVC/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/SVC/php/basics.ipynb">✓</a></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.svm.NuSVC.html">sklearn.svm.NuSVC</a></td>
            <td align="center"><a href="examples/classifier/NuSVC/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/NuSVC/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/NuSVC/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/NuSVC/php/basics.ipynb">✓</a></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.svm.LinearSVC.html">sklearn.svm.LinearSVC</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/c/basics.ipynb">✓</a> <a href="examples/classifier/LinearSVC/c/compilation.py#L14">✓</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/js/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/go/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/php/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/ruby/basics.ipynb">✓</a></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.tree.DecisionTreeClassifier.html">sklearn.tree.DecisionTreeClassifier</a></td>
            <td align="center"><a href="examples/classifier/DecisionTreeClassifier/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/DecisionTreeClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/DecisionTreeClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/DecisionTreeClassifier/php/basics.ipynb">✓</a></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.RandomForestClassifier.html">sklearn.ensemble.RandomForestClassifier</a></td>
            <td align="center"><a href="examples/classifier/RandomForestClassifier/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/RandomForestClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/RandomForestClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">sklearn.ensemble.ExtraTreesClassifier</a></td>
            <td align="center"><a href="examples/classifier/ExtraTreesClassifier/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/ExtraTreesClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/ExtraTreesClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">sklearn.ensemble.AdaBoostClassifier</a></td>
            <td align="center"><a href="examples/classifier/AdaBoostClassifier/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/AdaBoostClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/AdaBoostClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">sklearn.neighbors.KNeighborsClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/KNeighborsClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/classifier/KNeighborsClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.neural_network.MLPClassifier.html">sklearn.neural_network.MLPClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/MLPClassifier/java/basics.ipynb">○</a></td>
            <td align="center"><a href="examples/classifier/MLPClassifier/js/basics.ipynb">○</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB">sklearn.naive_bayes.GaussianNB</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/GaussianNB/java/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB">sklearn.naive_bayes.BernoulliNB</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/BernoulliNB/java/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="left" width="40%">Regression</td>
            <td colspan="6" width="10%"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html">sklearn.neural_network.MLPRegressor</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"><a href="examples/regressor/MLPRegressor/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>

✓ = is full-featured, ○ = has minor exceptions

## Installation

```sh
pip install sklearn-porter
```

If you want the latest bleeding edge changes, you can install the module from the master (development) branch:

```sh
pip uninstall -y sklearn-porter
pip install --no-cache-dir https://github.com/nok/sklearn-porter/zipball/master
```

## Minimum requirements

```
- python>=2.7.3
- scikit-learn>=0.14.1
```

If you want to transpile a multilayer perceptron (<a href="http://scikit-learn.org/0.18/modules/generated/sklearn.neural_network.MLPClassifier.html">sklearn.neural_network.MLPClassifier</a>), you have to upgrade the scikit-learn package:

```
- scikit-learn>=0.18.0
```


## Usage

### Export

The following example shows how you can port a [decision tree model](http://scikit-learn.org/stable/modules/tree.html#classification) to Java:

```python
from sklearn.datasets import load_iris
from sklearn.tree import tree
from sklearn_porter import Porter

# Load data and train the classifier:
iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Export:
porter = Porter(clf, language='java')
output = porter.export()
print(output)
```

The exported [result](examples/classifier/DecisionTreeClassifier/java/basics.py#L18-L98) matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the decision tree.

### Prediction

Run the prediction(s) in the target programming language directly:

```python
# ...

# Prediction(s):
porter = Porter(clf, language='java')
Y_preds = porter.predict(X)
y_pred = porter.predict(X[0])
y_pred = porter.predict([1., 2., 3., 4.])
```

### Accuracy

Test the accuracy between the original and the ported estimator:

```python
# ...

# Accuracy:
porter = Porter(clf, language='java')
accuracy = porter.predict_test(X)
print(accuracy) # 1.0
```

### Command-line interface

This example shows how you can port a model from the command line. First of all you have to store the model to the [pickle format](http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example):

```python
# ...

# Extract estimator:
joblib.dump(clf, 'model.pkl')
```

After that the model can be transpiled by using the following command:

```sh
python -m sklearn_porter --input <pickle_file> [--output <destination_dir>] [--language {c,go,java,js,php,ruby}]
python -m sklearn_porter -i <pickle_file> [-o <destination_dir>] [-l {c,go,java,js,php,ruby}]
```

The following commands have all the same result:

```sh
python -m sklearn_porter --input model.pkl --language java
python -m sklearn_porter -i model.pkl -l java
```

By changing the language parameter you can set the target programming language:

```
python -m sklearn_porter -i model.pkl -l c
python -m sklearn_porter -i model.pkl -l go
python -m sklearn_porter -i model.pkl -l java
python -m sklearn_porter -i model.pkl -l js
python -m sklearn_porter -i model.pkl -l php
python -m sklearn_porter -i model.pkl -l ruby
```

Further information will be shown by using the `--help` parameter:

```sh
python -m sklearn_porter --help
python -m sklearn_porter -h
```


## Development

### Environment

Install the required [environment modules](environment.yml) by executing the script [environment.sh](recipes/environment.sh):

```sh
./recipes/environment.sh
```

```sh
conda config --add channels conda-forge
conda env create -n sklearn-porter python=2 -f environment.yml
source activate sklearn-porter
```

Furthermore [Node.js](https://nodejs.org) (`>=6`), [Java](https://java.com) (`>=1.6`), [PHP](http://www.php.net/) (`>=7`), [Ruby](https://www.ruby-lang.org) (`>=1.9.3`) and [GCC](https://gcc.gnu.org) (`>=4.2`) are required for all tests.


### Testing

The tests cover module functions as well as matching predictions of transpiled models. Run [all tests](tests) by executing the script [test.sh](recipes/test.sh):

```sh
./recipes/test.sh
```

```sh
source activate sklearn-porter
python -m unittest discover -vp '*Test.py'
source deactivate
```

While you are developing new features or fixes, you can reduce the test duration by setting the number of random model tests:

```
N_RANDOM_TESTS=30 python -m unittest discover -vp '*Test.py'
```


### Quality

It's highly recommended to ensure the code quality. For that I use [Pylint](https://github.com/PyCQA/pylint/), which you can run by executing the script [lint.sh](recipes/lint.sh): 

```sh
./recipes/lint.sh
```

```sh
source activate sklearn-porter
find ./sklearn_porter -name '*.py' -exec pylint {} \;
source deactivate
```


## Questions?

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/sklearn-porter).


## License

The module is Open Source Software released under the [MIT](license.txt) license.