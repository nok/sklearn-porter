from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
base_estimator = DecisionTreeClassifier(max_depth=4, random_state=0)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100,
                         random_state=0)
clf.fit(X, y)

# Cheese!

result = Porter(language='js').port(clf)
print(result)

"""
var predictor = function(atts) {

    var predict_0 = function(atts) {
        var classes = new Array(3);

        if (atts[3] <= 0.80000001192092896) {
            classes[0] = 0.33333333333333298;
            classes[1] = 0.0;
            classes[2] = 0.0;
        } else {
            if (atts[3] <= 1.75) {
                if (atts[2] <= 4.9499998092651367) {
                    if (atts[3] <= 1.6500000953674316) {
                        classes[0] = 0.0;
                        classes[1] = 0.31333333333333302;
                        classes[2] = 0.0;
                    } else {
                        classes[0] = 0.0;
                        classes[1] = 0.0;
                        classes[2] = 0.0066666666666666671;
                    }
                } else {
                    if (atts[3] <= 1.5499999523162842) {
                        classes[0] = 0.0;
                        classes[1] = 0.0;
                        classes[2] = 0.02;
                    } else {
                        classes[0] = 0.0;
                        classes[1] = 0.013333333333333334;
                        classes[2] = 0.0066666666666666671;
                    }
                }
            } else {
                if (atts[2] <= 4.8500003814697266) {
                    if (atts[0] <= 5.9499998092651367) {
                        classes[0] = 0.0;
                        classes[1] = 0.0066666666666666671;
                        classes[2] = 0.0;
                    } else {
                        classes[0] = 0.0;
                        classes[1] = 0.0;
                        classes[2] = 0.013333333333333334;
                    }
                } else {
                    classes[0] = 0.0;
                    classes[1] = 0.0;
                    classes[2] = 0.2866666666666664;
                }
            }
        }
        return classes;
    };
    var predict_1 = function(atts) {
        var classes = new Array(3);

        if (atts[2] <= 5.1499996185302734) {
            if (atts[2] <= 2.4500000476837158) {
                classes[0] = 8.3290724464028397e-05;
                classes[1] = 0.0;
                classes[2] = 0.0;
            } else {
                if (atts[3] <= 1.75) {
                    if (atts[0] <= 4.9499998092651367) {
                        classes[0] = 0.0;
                        classes[1] = 1.6658144892805682e-06;
                        classes[2] = 1.6658144892805682e-06;
                    } else {
                        classes[0] = 0.0;
                        classes[1] = 0.4999541901015449;
                        classes[2] = 3.3316289785611363e-06;
                    }
                } else {
                    if (atts[1] <= 3.1500000953674316) {
                        classes[0] = 0.0;
                        classes[1] = 0.0;
                        classes[2] = 1.9989773871366814e-05;
                    } else {
                        classes[0] = 0.0;
                        classes[1] = 1.6658144892805682e-06;
                        classes[2] = 1.6658144892805682e-06;
                    }
                }
            }
        } else {
            classes[0] = 0.0;
            classes[1] = 0.0;
            classes[2] = 0.4999325345131842;
        }
        return classes;
    };
    var predict_2 = function(atts) {
        var classes = new Array(3);

        if (atts[3] <= 1.5499999523162842) {
            if (atts[2] <= 4.9499998092651367) {
                if (atts[3] <= 0.80000001192092896) {
                    classes[0] = 2.6788177186451792e-08;
                    classes[1] = 0.0;
                    classes[2] = 0.0;
                } else {
                    classes[0] = 0.0;
                    classes[1] = 0.00018473109499329488;
                    classes[2] = 0.0;
                }
            } else {
                classes[0] = 0.0;
                classes[1] = 0.0;
                classes[2] = 0.49969664310232625;
            }
        } else {
            if (atts[2] <= 5.1499996185302734) {
                if (atts[3] <= 1.8499999046325684) {
                    if (atts[0] <= 5.4000000953674316) {
                        classes[0] = 0.0;
                        classes[1] = 0.0;
                        classes[2] = 0.00011147301524887026;
                    } else {
                        classes[0] = 0.0;
                        classes[1] = 0.49973485750206614;
                        classes[2] = 2.6788177186451756e-09;
                    }
                } else {
                    classes[0] = 0.0;
                    classes[1] = 0.0;
                    classes[2] = 0.00011147676559367639;
                }
            } else {
                classes[0] = 0.0;
                classes[1] = 0.0;
                classes[2] = 0.00016078905277695348;
            }
        }
        return classes;
    };
    var predict_3 = function(atts) {
        var classes = new Array(3);

        if (atts[3] <= 1.75) {
            if (atts[3] <= 1.5499999523162842) {
                if (atts[2] <= 4.9499998092651367) {
                    if (atts[3] <= 0.80000001192092896) {
                        classes[0] = 9.2576539737627342e-11;
                        classes[1] = 0.0;
                        classes[2] = 0.0;
                    } else {
                        classes[0] = 0.0;
                        classes[1] = 6.384072136521275e-07;
                        classes[2] = 0.0;
                    }
                } else {
                    classes[0] = 0.0;
                    classes[1] = 0.0;
                    classes[2] = 0.0017268881646907192;
                }
            } else {
                if (atts[0] <= 6.9499998092651367) {
                    if (atts[1] <= 2.5999999046325684) {
                        classes[0] = 0.0;
                        classes[1] = 0.0;
                        classes[2] = 3.8523658978481932e-07;
                    } else {
                        classes[0] = 0.0;
                        classes[1] = 0.49902423425502029;
                        classes[2] = 0.0;
                    }
                } else {
                    classes[0] = 0.0;
                    classes[1] = 0.0;
                    classes[2] = 5.5560730608384753e-07;
                }
            }
        } else {
            if (atts[1] <= 3.1500000953674316) {
                classes[0] = 0.0;
                classes[1] = 0.0;
                classes[2] = 0.49913557364140265;
            } else {
                if (atts[2] <= 4.9499998092651367) {
                    classes[0] = 0.0;
                    classes[1] = 0.00011133933639195673;
                    classes[2] = 0.0;
                } else {
                    classes[0] = 0.0;
                    classes[1] = 0.0;
                    classes[2] = 3.8525880815435657e-07;
                }
            }
        }
        return classes;
    };
    var predict = function(atts) {
        var n_estimators = 4,
            preds = new Array(n_estimators),
            n_classes = 3,
            classes = new Array(n_classes),
            normalizer, sum, idx, val,
            i, j;

        preds[0] = predict_0(atts);
        preds[1] = predict_1(atts);
        preds[2] = predict_2(atts);
        preds[3] = predict_3(atts);

        for (i = 0; i < n_estimators; i++) {
            normalizer = 0.;
            for (j = 0; j < n_classes; j++) {
                normalizer += preds[i][j];
            }
            if (normalizer == 0.) {
                normalizer = 1.0;
            }
            for (j = 0; j < n_classes; j++) {
                preds[i][j] = preds[i][j] / normalizer;
                if (preds[i][j] < 2.2250738585072014e-308) {
                    preds[i][j] = 2.2250738585072014e-308;
                }
                preds[i][j] = Math.log(preds[i][j]);
            }
            sum = 0.0;
            for (j = 0; j < n_classes; j++) {
                sum += preds[i][j];
            }
            for (j = 0; j < n_classes; j++) {
                preds[i][j] = (n_classes - 1) * (preds[i][j] - (1. / n_classes) * sum);
            }
        }
        for (i = 0; i < n_classes; i++) {
            classes[i] = 0.0;
        }
        for (i = 0; i < n_estimators; i++) {
            for (j = 0; j < n_classes; j++) {
                classes[j] += preds[i][j];
            }
        }
        idx = -1;
        val = Number.NEGATIVE_INFINITY;
        for (i = 0; i < n_classes; i++) {
            if (classes[i] > val) {
                idx = i;
                val = classes[i];
            }
        }
        return idx;
    };

    return predict(atts);
};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 == 4) {
        var argv = process.argv.slice(2);
        var prediction = predictor(argv);
        console.log(prediction);
    }
}
"""
