# Model Porting

Library to port trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) models to a low-level programming language like C or Java. It's recommended for limited embedded systems and critical applications where performance matters most.

**Please note that this project is under active development.**


## Target output algorithms

- Classification
- ~~Regression~~


## Target models

Either you can port a single [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) or a [~~AdaBoostClassifier~~](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) based on a set of pruned decision trees.

- [sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [~~sklearn.ensemble.AdaBoostClassifier~~](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- ... and more to be defined.


## Target programming languages

- Java
- ~~C~~


## Usage

### Package

In this example we extend the [official user guide example](http://scikit-learn.org/stable/modules/tree.html#classification):

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris

from onl.nok.sklearn.export.Export import Export

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

tree = Export.export(clf)
print(tree)
```

The resulted output matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the model:

```java
class Tmp {
    public static int predict(float[] atts) {
        int n_classes = 3;
        int[] classes = new int[n_classes];

        if (atts[3] <= 0.800000f) {
            classes[0] = 50;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[3] <= 1.750000f) {
                if (atts[2] <= 4.950000f) {
                    if (atts[3] <= 1.650000f) {
                        classes[0] = 0;
                        classes[1] = 47;
                        classes[2] = 0;
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 1;
                    }
                } else {
                    if (atts[3] <= 1.550000f) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 3;
                    } else {
                        if (atts[2] <= 5.450000f) {
                            classes[0] = 0;
                            classes[1] = 2;
                            classes[2] = 0;
                        } else {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 1;
                        }
                    }
                }
            } else {
                if (atts[2] <= 4.850000f) {
                    if (atts[1] <= 3.100000f) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 2;
                    } else {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                    }
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 43;
                }
            }
        }

        int idx = 0;
        int val = classes[0];
        for (int i = 1; i < n_classes; i++) {
            if (classes[i] > val) {
                idx = i;
            }
        }
        return idx;
    }

    public static void main(String[] args) {
        if (args.length == 4) {
            float[] atts = new float[args.length];
            for (int i = 0; i < args.length; i++) {
                atts[i] = Float.parseFloat(args[i]);
            }
            System.out.println(Tmp.predict(atts));
        }
    }
}
```

### CLI

Alternatively we can save the model on the file system:

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn.externals import joblib

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

joblib.dump(clf, 'model.pkl')
```
After that we can port the dumped model on the command line:

```sh
python Export.py model.pkl Model.java
```


## Environment

Install the [environment modules](environment.yml) by executing the bash script [environment.sh](environment.sh) or typing:

```sh
conda env create -f environment.yml
```

## Unit testing

Run the [tests](tests) by executing the bash script [tests.sh](tests.sh) or typing:

```sh
python -m unittest discover -p '*Test.py'
```


## Questions?

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/hi).


## License

The library is Open Source Software released under the [license](LICENSE.txt).