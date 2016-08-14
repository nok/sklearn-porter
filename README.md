[![Build Status](https://img.shields.io/travis/nok/scikit-learn-model-porting/master.svg)](https://travis-ci.org/nok/scikit-learn-model-porting) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/nok/scikit-learn-model-porting/master/LICENSE.txt)

---

# Model Porting

Library to port trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) models to a low-level programming language like C or Java. It's recommended for limited embedded systems and critical applications where performance matters most.

**Please note that this project is under active development.**


## Target output algorithms

- Classification
- ~~Regression~~


## Target models

Either you can port a single [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) or a [AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) based on a set of pruned decision trees.

- [sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [sklearn.ensemble.AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)


## Target programming languages

- Java
- ~~C~~
- ~~JavaScript~~


## Usage

### sklearn.tree.DecisionTreeClassifier

#### Package

In this example we extend the [official user guide example](http://scikit-learn.org/stable/modules/tree.html#classification):

```python
from sklearn.tree import tree
from sklearn.datasets import load_iris

from onl.nok.sklearn.export import export

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

tree = export(clf)
print(tree)
```

The resulted output matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the model:

```java
class Tmp {
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
                val = classes[i];
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

#### CLI

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

### sklearn.ensemble.AdaBoostClassifier

#### Package

In this example we use multiple decision trees:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from onl.nok.sklearn.export import export

iris = load_iris()
base_estimator = DecisionTreeClassifier(max_depth=4)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100)
clf.fit(iris.data, iris.target)

trees = export(clf)

print(trees)
```

```java
class Tmp {
    public static float[] predict_000(float[] atts) {
        int n_classes = 3;
        float[] classes = new float[n_classes];

        if (atts[3] <= 0.800000011920929f) {
            classes[0] = 0.333333333333333f;
            classes[1] = 0.0f;
            classes[2] = 0.0f;
        } else {
            if (atts[3] <= 1.75f) {
                if (atts[2] <= 4.949999809265137f) {
                    if (atts[3] <= 1.650000095367432f) {
                        classes[0] = 0.0f;
                        classes[1] = 0.313333333333333f;
                        classes[2] = 0.0f;
                    } else {
                        classes[0] = 0.0f;
                        classes[1] = 0.0f;
                        classes[2] = 0.006666666666666667f;
                    }
                } else {
                    if (atts[3] <= 1.549999952316284f) {
                        classes[0] = 0.0f;
                        classes[1] = 0.0f;
                        classes[2] = 0.02f;
                    } else {
                        classes[0] = 0.0f;
                        classes[1] = 0.01333333333333333f;
                        classes[2] = 0.006666666666666667f;
                    }
                }
            } else {
                if (atts[2] <= 4.850000381469727f) {
                    if (atts[0] <= 5.949999809265137f) {
                        classes[0] = 0.0f;
                        classes[1] = 0.006666666666666667f;
                        classes[2] = 0.0f;
                    } else {
                        classes[0] = 0.0f;
                        classes[1] = 0.0f;
                        classes[2] = 0.01333333333333333f;
                    }
                } else {
                    classes[0] = 0.0f;
                    classes[1] = 0.0f;
                    classes[2] = 0.2866666666666664f;
                }
            }
        }

        return classes;
    }

    public static float[] predict_001(float[] atts) {
        int n_classes = 3;
        float[] classes = new float[n_classes];

        if (atts[2] <= 5.149999618530273f) {
            if (atts[2] <= 2.450000047683716f) {
                classes[0] = 8.32907244640284e-05f;
                classes[1] = 0.0f;
                classes[2] = 0.0f;
            } else {
                if (atts[3] <= 1.75f) {
                    if (atts[0] <= 4.949999809265137f) {
                        classes[0] = 0.0f;
                        classes[1] = 1.665814489280568e-06f;
                        classes[2] = 1.665814489280568e-06f;
                    } else {
                        classes[0] = 0.0f;
                        classes[1] = 0.499954190101545f;
                        classes[2] = 3.331628978561136e-06f;
                    }
                } else {
                    if (atts[1] <= 3.150000095367432f) {
                        classes[0] = 0.0f;
                        classes[1] = 0.0f;
                        classes[2] = 1.998977387136681e-05f;
                    } else {
                        classes[0] = 0.0f;
                        classes[1] = 1.665814489280568e-06f;
                        classes[2] = 1.665814489280568e-06f;
                    }
                }
            }
        } else {
            classes[0] = 0.0f;
            classes[1] = 0.0f;
            classes[2] = 0.4999325345131842f;
        }

        return classes;
    }

    public static float[] predict_002(float[] atts) {
        int n_classes = 3;
        float[] classes = new float[n_classes];

        if (atts[3] <= 1.549999952316284f) {
            if (atts[2] <= 4.949999809265137f) {
                if (atts[3] <= 0.800000011920929f) {
                    classes[0] = 2.678817718645179e-08f;
                    classes[1] = 0.0f;
                    classes[2] = 0.0f;
                } else {
                    classes[0] = 0.0f;
                    classes[1] = 0.0001847310949932946f;
                    classes[2] = 0.0f;
                }
            } else {
                classes[0] = 0.0f;
                classes[1] = 0.0f;
                classes[2] = 0.4996966431023263f;
            }
        } else {
            if (atts[2] <= 5.149999618530273f) {
                if (atts[3] <= 1.849999904632568f) {
                    if (atts[1] <= 2.599999904632568f) {
                        classes[0] = 0.0f;
                        classes[1] = 0.0f;
                        classes[2] = 0.0001114730152488703f;
                    } else {
                        classes[0] = 0.0f;
                        classes[1] = 0.4997348575020662f;
                        classes[2] = 2.678817718645176e-09f;
                    }
                } else {
                    classes[0] = 0.0f;
                    classes[1] = 0.0f;
                    classes[2] = 0.0001114767655936764f;
                }
            } else {
                classes[0] = 0.0f;
                classes[1] = 0.0f;
                classes[2] = 0.0001607890527769535f;
            }
        }

        return classes;
    }

    public static float[] predict_003(float[] atts) {
        int n_classes = 3;
        float[] classes = new float[n_classes];

        if (atts[3] <= 1.75f) {
            if (atts[3] <= 1.549999952316284f) {
                if (atts[2] <= 4.949999809265137f) {
                    if (atts[2] <= 2.450000047683716f) {
                        classes[0] = 9.257653973762734e-11f;
                        classes[1] = 0.0f;
                        classes[2] = 0.0f;
                    } else {
                        classes[0] = 0.0f;
                        classes[1] = 6.384072136521276e-07f;
                        classes[2] = 0.0f;
                    }
                } else {
                    classes[0] = 0.0f;
                    classes[1] = 0.0f;
                    classes[2] = 0.001726888164690719f;
                }
            } else {
                if (atts[0] <= 6.949999809265137f) {
                    if (atts[1] <= 2.599999904632568f) {
                        classes[0] = 0.0f;
                        classes[1] = 0.0f;
                        classes[2] = 3.852365897848193e-07f;
                    } else {
                        classes[0] = 0.0f;
                        classes[1] = 0.4990242342550203f;
                        classes[2] = 0.0f;
                    }
                } else {
                    classes[0] = 0.0f;
                    classes[1] = 0.0f;
                    classes[2] = 5.556073060838475e-07f;
                }
            }
        } else {
            if (atts[1] <= 3.150000095367432f) {
                classes[0] = 0.0f;
                classes[1] = 0.0f;
                classes[2] = 0.4991355736414027f;
            } else {
                if (atts[2] <= 4.949999809265137f) {
                    classes[0] = 0.0f;
                    classes[1] = 0.0001113393363919567f;
                    classes[2] = 0.0f;
                } else {
                    classes[0] = 0.0f;
                    classes[1] = 0.0f;
                    classes[2] = 3.852588081543566e-07f;
                }
            }
        }

        return classes;
    }

    public static int predict(float[] atts) {
        int n_estimators = 4;
        int n_classes = 3;

        float[][] preds = new float[n_estimators][];
        preds[0] = Tmp.predict_000(atts);
        preds[1] = Tmp.predict_001(atts);
        preds[2] = Tmp.predict_002(atts);
        preds[3] = Tmp.predict_003(atts);

        int i, j;
        float normalizer, sum;
        for (i = 0; i < n_estimators; i++) {
            normalizer = 0.f;
            for (j = 0; j < n_classes; j++) {
                normalizer += preds[i][j];
            }
            if (normalizer == 0.f) {
                normalizer = 1.0f;
            }
            for (j = 0; j < n_classes; j++) {
                preds[i][j] = preds[i][j] / normalizer;
                if (preds[i][j] < 0.000000000000000222044604925f) {
                    preds[i][j] = 0.000000000000000222044604925f;
                }
                preds[i][j] = (float) Math.log(preds[i][j]);
            }
            sum = 0.0f;
            for (j = 0; j < n_classes; j++) {
                sum += preds[i][j];
            }
            for (j = 0; j < n_classes; j++) {
                preds[i][j] = (n_classes - 1) * (preds[i][j] - (1.f / n_classes) * sum);
            }
        }
        float[] classes = new float[n_classes];
        for (i = 0; i < n_estimators; i++) {
            for (j = 0; j < n_classes; j++) {
                classes[j] += preds[i][j];
            }
        }
        int idx = 0;
        float val = Float.NEGATIVE_INFINITY;
        for (i = 0; i < n_classes; i++) {
            if (classes[i] > val) {
                idx = i;
                val = classes[i];
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

## Environment

Install the [environment modules](environment.yml) by executing the bash script [environment.sh](environment.sh) or typing:

```sh
conda config --add channels conda-forge
conda env create -n sklearn.tree.model.export python=2 -f environment.yml
source activate sklearn.tree.model.export
```

## Unit testing

Run the [tests](tests) by executing the bash script [tests.sh](tests.sh) or typing:

```sh
python -m unittest discover -p '*Test.py'
```

```sh
python -m unittest discover -v -p '*Test.py'
```


## Questions?

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/hi).


## License

The library is Open Source Software released under the [license](LICENSE.txt).