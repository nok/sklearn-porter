
# sklearn-porter

[![Build Status](https://img.shields.io/travis/nok/sklearn-porter/master.svg)](https://travis-ci.org/nok/sklearn-porter)
[![Join the chat at https://gitter.im/nok/sklearn-porter](https://badges.gitter.im/nok/sklearn-porter.svg)](https://gitter.im/nok/sklearn-porter?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/nok/scikit-learn-model-porting/master/LICENSE.txt)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) models to a low-level programming language like C, Java or JavaScript. It's recommended for limited embedded systems and critical applications where performance matters most.


## Target Algorithm Models

### Classification

The following matrix shows the portable classifier models:

<table>
    <tbody>
        <tr>
            <td width="32%"><strong>Classifier</strong></td>
            <td align="center" width="17%"><strong>Example(s)</strong></td>
            <td align="center" width="17%"><strong>C</strong></td>
            <td align="center" width="17%"><strong>Java</strong></td>
            <td align="center" width="17%"><strong>JavaScript</strong></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.svm.SVC.html">sklearn.svm.SVC</a></td>
            <td align="center"><a href="examples/classifier/SVC/java_predict.py">X</a></td>
            <td align="center"></td>
            <td align="center">X</td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.svm.LinearSVC.html">sklearn.svm.LinearSVC</a></td>
            <td align="center"><a href="examples/classifier/LinearSVC/java_predict.py">X</a></td>
            <td align="center">X</td>
            <td align="center">X</td>
            <td align="center">X</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.tree.DecisionTreeClassifier.html">sklearn.tree.DecisionTreeClassifier</a></td>
            <td align="center"><a href="examples/classifier/DecisionTreeClassifier/java_predict.py">X</a></td>
            <td align="center"></td>
            <td align="center">X</td>
            <td align="center">X</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.RandomForestClassifier.html">sklearn.ensemble.RandomForestClassifier</a></td>
            <td align="center"><a href="examples/classifier/RandomForestClassifier/java_predict.py">X</a></td>
            <td align="center"></td>
            <td align="center">X</td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">sklearn.ensemble.ExtraTreesClassifier</a></td>
            <td align="center"><a href="examples/classifier/ExtraTreesClassifier/java_predict.py">X</a></td>
            <td align="center"></td>
            <td align="center">X</td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/0.18/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">sklearn.ensemble.AdaBoostClassifier</a></td>
            <td align="center"><a href="examples/classifier/AdaBoostClassifier/java_predict.py">X</a></td>
            <td align="center"></td>
            <td align="center">X</td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>

<!--
### ~~Regression~~
The following matrix shows the portable regression models:
-->


## Usage

Either you use the porter in your application as [imported module](#module) or you use it on the [command line](#cli). 


### Module

This example shows how you can port the decision tree model from the [official user guide](http://scikit-learn.org/0.17/modules/tree.html#classification) to the programming language Java:

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris

from onl.nok.sklearn.Porter import port

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

# Here we port the model:
tree = port(clf)
print(tree)
```

The [result](examples/classifier/decisiontree_predict.py) matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the model.


### CLI

This examples shows how you can port a model from the command line. First of all you have to store the model to the [pickle format](http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example):

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn.externals import joblib

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

joblib.dump(clf, 'model.pkl')
```

Then you can port the model by using the following command:

```sh
python onl/nok/sklearn/Porter.py --model <pickle_model_path> --output <output_file_path> [--language {c,java,js}]
```

Here are some examples:

```sh
python onl/nok/sklearn/Porter.py --model examples/cli/model.pkl --output examples/cli/Model.java --language java
python onl/nok/sklearn/Porter.py -m examples/cli/model.pkl -o examples/cli/Model.java -l java
```

For further help you can display all options:

```sh
python onl/nok/sklearn/Porter.py --help
python onl/nok/sklearn/Porter.py -h
```


## Environment

Install the [environment modules](environment.yml) by executing the bash script [environment.sh](environment.sh) or type:

```sh
conda config --add channels conda-forge
conda env create -n sklearn-porter python=2 -f environment.yml
source activate sklearn-porter
```

## Unit Testing

Run all [tests](tests) by executing the bash script [tests.sh](tests.sh) or type:

```sh
python -m unittest discover -vp '*Test.py'
```


## Questions?

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/sklearn-porter).


## License

The library is Open Source Software released under the [MIT](license.txt) license.