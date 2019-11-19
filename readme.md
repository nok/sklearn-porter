
# sklearn-porter

[![GitHub license](https://img.shields.io/pypi/l/sklearn-porter.svg)](https://raw.githubusercontent.com/nok/sklearn-porter/master/license.txt)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nok/sklearn-porter/feature/oop-api-refactoring?filepath=examples)
[![Stack Overflow](https://img.shields.io/badge/stack%20overflow-ask%20questions-blue.svg)](https://stackoverflow.com/questions/tagged/sklearn-porter)
[![Join the chat at https://gitter.im/nok/sklearn-porter](https://badges.gitter.im/nok/sklearn-porter.svg)](https://gitter.im/nok/sklearn-porter?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Twitter](https://img.shields.io/twitter/follow/darius_morawiec.svg?label=follow&style=popout)](https://twitter.com/darius_morawiec)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators to C, Java, JavaScript and others.<br>It's recommended for limited embedded systems and critical applications where performance matters most.


## Important

We're hard working on the [first major release](https://github.com/nok/sklearn-porter/tree/release/1.0.0) of sklearn-porter. <br>Until that we will just release bugfixes to the stable version.


## Estimators

<table>
  <tr>
    <th></th>
    <th colspan="18">Programming language</th>
  </tr>
  <tr align="center">
    <th></th>
    <th colspan="3">C</th>
    <th colspan="3">Go</th>
    <th colspan="3">Java</th>
    <th colspan="3">JS</th>
    <th colspan="3">PHP</th>
    <th colspan="3">Ruby</th>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">svm.SVC</a>
    </td>
    <td>✓</td>
    <td></td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html">svm.NuSVC</a>
    </td>
    <td>✓</td>
    <td></td>
    <td>×</td>
    <td>✓</td>
    <td></td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">svm.LinearSVC</a>
    </td>
    <td>✓</td>
    <td></td>
    <td>×</td>
    <td>✓</td>
    <td></td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>×</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">tree.DecisionTreeClassifier</a>
    </td>
    <td>✓ᴾ</td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">ensemble.RandomForestClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">ensemble.ExtraTreesClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">ensemble.AdaBoostClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">neighbors.KNeighborsClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB">naive_bayes.BernoulliNB</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB">naive_bayes.GaussianNB</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">neural_network.MLPClassifier</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td>✓ᴾ</td>
    <td>✓ᴾ</td>
    <td>×</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td align="left">
      <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html">neural_network.MLPRegressor</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr align="center">
    <td></td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
    <td>ᴀ</td>
    <td>ᴇ</td>
    <td>ᴄ</td>
  </tr>
  <tr>
    <th></th>
    <th colspan="18">Template</th>
  </tr>
</table>

✓ = support of `predict`,　ᴾ = support of `predict_proba`,　× = not supported or feasible<br>
ᴀ = attached model data,　ᴇ = exported model data (JSON),　ᴄ = combined model data (not recommended)

---

<table>
    <tbody>
        <tr>
            <td align="center" width="32%"><strong>Estimator</strong></td>
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
            <td align="center" width="13%">Java *</td>
            <td align="center" width="11%">JS</td>
            <td align="center" width="11%">C</td>
            <td align="center" width="11%">Go</td>
            <td align="center" width="11%">PHP</td>
            <td align="center" width="11%">Ruby</td>
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

### API

The following table shows the most relevant high-level methods. 

<table>
  <tr>
    <th align="left">Step</th>
    <th align="left">Method</th>
    <th align="left">Alias</th>
    <th align="left">Description</th>
  </tr>
  <tr>
    <td>1</td>
    <td><code>port</code></td>
    <td><code>export</code></td>
    <td>Transpile the passed estimator to the desired programming language and template.</td>
  </tr>
  <tr>
    <td>2</td>
    <td><code>save</code></td>
    <td><code>dump</code></td>
    <td>Save the previously generated source code locally.</td>
  </tr>
  <tr>
    <td>3</td>
    <td><code>make</code></td>
    <td><code>predict</code></td>
    <td>Compile the saved source files and make predictions.</td>
  </tr>
  <tr>
    <td>4</td>
    <td><code>test</code></td>
    <td><code>integrity_score</code></td>
    <td>Make an integrity check by making regression tests between the original and transpiled estimator.</td>
  </tr>
</table>

Each step executes the previous steps internally.


### Examples

### Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nok/sklearn-porter/feature/oop-api-refactoring?filepath=examples)

Start Binder to run interactive examples.

#### Export

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


#### Integrity

You should always check and compute the integrity between the original and the transpiled estimator:

```python
# ...
porter = Porter(clf, language='java')

# Compute integrity score:
integrity = porter.integrity_score(X)
print(integrity)  # 1.0
```


#### Prediction

You can compute the prediction(s) in the target programming language:

```python
# ...
porter = Porter(clf, language='java')

# Prediction(s):
Y_java = porter.predict(X)
y_java = porter.predict(X[0])
y_java = porter.predict([1., 2., 3., 4.])
```


## Notebooks

You can run and test all notebooks by starting a Jupyter notebook server locally:

```bash
$ make open.examples
$ make stop.examples
```


## CLI

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


### Dependencies

The prerequisite is Python 3.5 which you can install with [conda](https://docs.conda.io/en/latest/miniconda.html):

```bash
$ conda env create -n sklearn-porter -c defaults python=3.5
$ conda activate sklearn-porter  # or `source activate sklearn-porter` for older versions
```

After that you have to install all required packages:

```bash
$ pip install --no-cache-dir -e .[development]
```

### Environment

All tests run against these combinations of [scikit-learn](https://github.com/scikit-learn/scikit-learn) and Python versions:

<table border="0" width="100%">
  <tr align="center">
    <td colspan="2" rowspan="2"></td>
    <td colspan="3"><strong>Python</strong></td>
  </tr>
  <tr align="center">
    <td><strong>3.5</strong></td>
    <td><strong>3.6</strong></td>
    <td><strong>3.7</strong></td>
  </tr>
  <tr align="center">
    <td rowspan="15"><strong>scikit-learn</strong></td>
    <td rowspan="3"><strong>0.17</strong></td>
    <td>cython 0.27.3</td>
    <td>cython 0.27.3</td>
    <td rowspan="3">not supported<br>by scikit-learn</td>
  </tr>
  <tr align="center">
    <td>numpy 1.9.3</td>
    <td>numpy 1.9.3</td>
  </tr>
  <tr align="center">
    <td>scipy 0.16.0</td>
    <td>scipy 0.16.0</td>
  </tr>
  <tr align="center">
    <td rowspan="3"><strong>0.18</strong></td>
    <td>cython 0.27.3</td>
    <td>cython 0.27.3</td>
    <td rowspan="3">not supported<br>by scikit-learn</td>
  </tr>
  <tr align="center">
    <td>numpy 1.9.3</td>
    <td>numpy 1.9.3</td>
  </tr>
  <tr align="center">
    <td>scipy 0.16.0</td>
    <td>scipy 0.16.0</td>
  </tr>
  <tr align="center">
    <td rowspan="3"><strong>0.19</strong></td>
    <td>cython 0.27.3</td>
    <td>cython 0.27.3</td>
    <td rowspan="3">not supported<br>by scikit-learn</td>
  </tr>
  <tr align="center">
    <td>numpy 1.14.5</td>
    <td>numpy 1.14.5</td>
  </tr>
  <tr align="center">
    <td>scipy 1.1.0</td>
    <td>scipy 1.1.0</td>
  </tr>
  <tr align="center">
    <td rowspan="3"><strong>0.20</strong></td>
    <td>cython 0.27.3</td>
    <td>cython 0.27.3</td>
    <td>cython 0.27.3</td>
  </tr>
  <tr align="center">
    <td>numpy</td>
    <td>numpy</td>
    <td>numpy</td>
  </tr>
  <tr align="center">
    <td>scipy</td>
    <td>scipy</td>
    <td>scipy</td>
  </tr>
  <tr align="center">
    <td rowspan="3"><strong>0.21</strong></td>
    <td>cython</td>
    <td>cython</td>
    <td>cython</td>
  </tr>
  <tr align="center">
    <td>numpy</td>
    <td>numpy</td>
    <td>numpy</td>
  </tr>
  <tr align="center">
    <td>scipy</td>
    <td>scipy</td>
    <td>scipy</td>
  </tr>
</table>

For the regression tests we have to use specific compilers and interpreters. On 19th November 2019 the following compilers and interpreters are used for these tests:

<table>
  <tr>
    <th align="left">Name</th>
    <th align="left">Source</th>
    <th align="left">Version</th>
  </tr>
  <tr>
    <td>GCC</td>
    <td><a href="https://gcc.gnu.org">https://gcc.gnu.org</a></td>
    <td>9.2.1</td>
  </tr>
  <tr>
    <td>Go</td>
    <td><a href="https://golang.org">https://golang.org</a></td>
    <td>1.13.4</td>
  </tr>
  <tr>
    <td>Java (OpenJDK)</td>
    <td><a href="https://openjdk.java.net">https://openjdk.java.net</a></td>
    <td>1.8.0</td>
  </tr>
  <tr>
    <td>Node.js</td>
    <td><a href="https://nodejs.org/en/">https://nodejs.org</a></td>
    <td>10.17.0</td>
  </tr>
  <tr>
    <td>PHP</td>
    <td><a href="https://www.php.net/">https://www.php.net</a></td>
    <td>7.3.11</td>
  </tr>
  <tr>
    <td>Ruby</td>
    <td><a href="https://www.ruby-lang.org/en/">https://www.ruby-lang.org</a></td>
    <td>2.5.7</td>
  </tr>
</table>

Please notice that in general you can use older compilers and interpreters with the generated source code. For instance you can use Java 1.6 to compile and run models.

### Logging

You can activate logging by changing the option `logging.level`.

```python
from sklearn_porter import options

from logging import DEBUG

options['logging.level'] = DEBUG
```

### Testing

The tests cover module functions as well as matching predictions of transpiled estimators. Start all tests with:

```bash
$ make tests
```

The test files have a specific pattern: `'[Algorithm][Language]Test.py'`:

```bash
$ make tests python_files="*JavaTest.py"
```


### Quality

It's highly recommended to ensure the code quality. For that [Pylint](https://github.com/PyCQA/pylint/) is used. Start the linter with:

```bash
$ make lint
```


## Citation

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

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/sklearn-porter).
