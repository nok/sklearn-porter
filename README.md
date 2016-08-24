
# sklearn-porter

[![Build Status](https://img.shields.io/travis/nok/sklearn-porter/master.svg)](https://travis-ci.org/nok/sklearn-porter)
[![Join the chat at https://gitter.im/nok/sklearn-porter](https://badges.gitter.im/nok/sklearn-porter.svg)](https://gitter.im/nok/sklearn-porter?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/nok/scikit-learn-model-porting/master/LICENSE.txt)

Module to port trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) models to a low-level programming language like C, Java or JavaScript. It's recommended for limited embedded systems and critical applications where performance matters most.


## Target Algorithm Models

### Classification

The following matrix shows the portable classifier models:

<table>
    <tbody>
        <tr>
            <td width="40%"></td>
            <td align="center" width="20%"><strong>C</strong></td>
            <td align="center" width="20%"><strong>Java</strong></td>
            <td align="center" width="20%"><strong>JavaScript</strong></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">DecisionTreeClassifier</a></td>
            <td align="center"></td>
            <td align="center">X</td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">AdaBoostClassifier</a></td>
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

This example shows how to port the [official user guide example](http://scikit-learn.org/stable/modules/tree.html#classification) to Java:

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris

from onl.nok.sklearn.Porter import port

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

tree = port(clf)
print(tree)
```

The [resulted output](examples/classifier/decisiontree_predict.py) matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the model.


### CLI

This examples shows how you can port a model from the command line. First of all you have to store the model in pickle format:

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn.externals import joblib

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

joblib.dump(clf, 'model.pkl')
```

Then you can port the model by typing the following command:

```sh
python Porter.py Model.java model.pkl
```

All options can be shown by running:

```sh
python Porter.py -h
```


## Environment

Install the [environment modules](environment.yml) by executing the bash script [environment.sh](environment.sh) or by typing:

```sh
conda config --add channels conda-forge
conda env create -n sklearn-porter python=2 -f environment.yml
source activate sklearn-porter
```

## Unit Testing

Run the [tests](tests) by executing the bash script [tests.sh](tests.sh) or by typing:

```sh
python -m unittest discover -vp '*Test.py'
```


## Questions?

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/sklearn-porter).


## License

The library is Open Source Software released under the [license](LICENSE.txt).