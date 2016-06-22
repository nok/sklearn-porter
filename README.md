# DecisionTree(s) Porting

Static methods to port learned decision tree models to a low-level programming language like C or Java. It's recommended for limited embedded systems and critical applications where performance matters most.

**Note: This project is still under active development.**

## Different variants

- [sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
	- predict()
	- ~~predict_proba()~~
- ~~[sklearn.ensemble.AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)~~
	- ~~predict()~~
	- ~~predict_proba()~~

## Target programming languages

- Java
- ~~C~~

## Usage

### Package

In this example we extend the [official user guide example](http://scikit-learn.org/stable/modules/tree.html#classification):

```
from sklearn.tree import tree
from sklearn.datasets import load_iris

from nok.Export import Export

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

tree = Export.predict(clf)
print(tree)
```

The resulted output matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the model:

```
public static int predict(float[] atts) {
    int n_classes = 3;
    int[] classes = new int[n_classes];

    if (atts[2] <= 2.450000f) {
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
                if (atts[0] <= 5.950000f) {
                    classes[0] = 0;
                    classes[1] = 1;
                    classes[2] = 0;
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 2;
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
```

### CLI

Alternatively we can save the model on the file system:

```
from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn.externals import joblib

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

joblib.dump(clf, 'model.pkl')
```
After that we can port the saved model on the command line:

```
python Export.py model.pkl Model.java
```


## Environment

Install the [environment modules](environment.yml) by executing the bash script [environment.sh](environment.sh) or typing `conda env create -f environment.yml`.


## Unit Testing

Run the [tests](tests) by executing the bash script [test.sh](test.sh) or typing `python -m unittest discover -p '*Test.py'`.


## Questions?

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/hi).


## License

The library is Open Source Software released under the [license](LICENSE.txt).