
# sklearn-porter

[![Build Status](https://img.shields.io/travis/nok/sklearn-porter/master.svg)](https://travis-ci.org/nok/sklearn-porter)
[![Join the chat at https://gitter.im/nok/sklearn-porter](https://badges.gitter.im/nok/sklearn-porter.svg)](https://gitter.im/nok/sklearn-porter?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/nok/scikit-learn-model-porting/master/LICENSE.txt)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) models to a low-level programming language like [C](https://en.wikipedia.org/wiki/C_(programming_language)), [Java](https://en.wikipedia.org/wiki/Java_(programming_language)), [JavaScript](https://en.wikipedia.org/wiki/JavaScript), [Go](https://en.wikipedia.org/wiki/Go_(programming_language)) or [Swift](https://en.wikipedia.org/wiki/Swift_(programming_language)). It's recommended for limited embedded systems and critical applications where performance matters most.


## Machine learning algorithms

### Classification

The following matrix shows the portable classifiers:

<table>
    <tbody>
        <tr>
            <td width="35%"><strong>Classifier</strong></td>
            <td align="center" width="13%"><strong>C</strong></td>
            <td align="center" width="13%"><strong>Java</strong></td>
            <td align="center" width="13%"><strong>JavaScript</strong></td>
            <td align="center" width="13%"><strong>Go</strong></td>
            <td align="center" width="13%"><strong>Swift</strong></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.svm.SVC.html">sklearn.svm.SVC</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/SVC/java/example.py#L12">X</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.svm.LinearSVC.html">sklearn.svm.LinearSVC</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/c/example.py#L12">X</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/java/example.py#L12">X</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/js/example.py#L12">X</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/go/example.py#L12">X</a></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.tree.DecisionTreeClassifier.html">sklearn.tree.DecisionTreeClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/DecisionTreeClassifier/java/example.py#L12">X</a></td>
            <td align="center"><a href="examples/classifier/DecisionTreeClassifier/js/example.py#L12">X</a></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.RandomForestClassifier.html">sklearn.ensemble.RandomForestClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/RandomForestClassifier/java/example.py#L13">X</a></td>
            <td align="center"><a href="examples/classifier/RandomForestClassifier/js/example.py#L13">X</a></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">sklearn.ensemble.ExtraTreesClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/ExtraTreesClassifier/java/example.py#L12">X</a></td>
            <td align="center"><a href="examples/classifier/ExtraTreesClassifier/js/example.py#L12">X</a></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">sklearn.ensemble.AdaBoostClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/AdaBoostClassifier/java/example.py#L15">X</a></td>
            <td align="center"><a href="examples/classifier/AdaBoostClassifier/js/example.py#L15">X</a></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.neural_network.MLPClassifier.html">sklearn.neural_network.MLPClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/classifier/MLPClassifier/java/example.py#L25">X</a></td>
            <td align="center"><a href="examples/classifier/MLPClassifier/js/example.py#L25">X</a></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>


## Usage

Either you use the porter as [imported module](#module) in your application or you use the [command-line interface](#cli). 


### Module

This example shows how you can port the decision tree model from the [official user guide](http://scikit-learn.org/stable/modules/tree.html#classification) to Java:

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris

from onl.nok.sklearn.Porter import port

# Load data and train a classifier:
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Port the model:
print(port(clf))
```

The ported [result](examples/classifier/DecisionTreeClassifier/java/example.py#L15-L93) matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the model.


### Command-line interface

This examples shows how you can port a model from the command line. First of all you have to store the model to the [pickle format](http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example):

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn.externals import joblib

X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

joblib.dump(clf, 'model.pkl')
```

After that you can port the model by using the following command:

```sh
python onl/nok/sklearn/Porter.py --model <pickle_model_path> --output <output_file_path> [--language {c,java,js}]
```

Here you see some real examples:

```sh
python onl/nok/sklearn/Porter.py --model examples/cli/model.pkl --output examples/cli/Model.java --language java
python onl/nok/sklearn/Porter.py -m examples/cli/model.pkl -o examples/cli/Model.java -l java
```

For further help you can display all options:

```sh
python onl/nok/sklearn/Porter.py --help
python onl/nok/sklearn/Porter.py -h
```


## Development

### Environment

Install the required environment [modules](environment.yml) by executing the bash script [sh_environment.sh](sh_environment.sh) or type:

```sh
conda config --add channels conda-forge
conda env create -n sklearn-porter python=2 -f environment.yml
```

Furthermore you need to install [Node.js](https://nodejs.org) (`>=6`), [Java](https://java.com) (`>=1.6`) and [GCC](https://gcc.gnu.org) (`>=4.2`) for testing.


### Testing

Run all [tests](tests) by executing the bash script [sh_tests.sh](sh_tests.sh) or type:

```sh
source activate sklearn-porter
python -m unittest discover -vp '*Test.py'
source deactivate
```

The tests cover module functions as well as matching predictions of ported models.


## Questions?

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/sklearn-porter).


## License

The library is Open Source Software released under the [MIT](license.txt) license.