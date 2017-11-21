# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

base_estimator = DecisionTreeClassifier(max_depth=4, random_state=0)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100,
                         random_state=0)
clf.fit(X, y)

porter = Porter(clf, language='js')
output = porter.export()
print(output)

"""
var AdaBoostClassifier = function() {

    var findMax = function(nums) {
        var index = 0;
        for (var i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    };

    var forest = new Array();

    forest.push(function(features) {
        var classes = new Array(3);
        if (features[3] <= 0.800000011921) {
            classes[0] = 0.333333333333; 
            classes[1] = 0.0; 
            classes[2] = 0.0; 
        } else {
            if (features[3] <= 1.75) {
                if (features[2] <= 4.94999980927) {
                    if (features[3] <= 1.65000009537) {
                        classes[0] = 0.0; 
                        classes[1] = 0.313333333333; 
                        classes[2] = 0.0; 
                    } else {
                        classes[0] = 0.0; 
                        classes[1] = 0.0; 
                        classes[2] = 0.00666666666667; 
                    }
                } else {
                    if (features[3] <= 1.54999995232) {
                        classes[0] = 0.0; 
                        classes[1] = 0.0; 
                        classes[2] = 0.02; 
                    } else {
                        classes[0] = 0.0; 
                        classes[1] = 0.0133333333333; 
                        classes[2] = 0.00666666666667; 
                    }
                }
            } else {
                if (features[2] <= 4.85000038147) {
                    if (features[0] <= 5.94999980927) {
                        classes[0] = 0.0; 
                        classes[1] = 0.00666666666667; 
                        classes[2] = 0.0; 
                    } else {
                        classes[0] = 0.0; 
                        classes[1] = 0.0; 
                        classes[2] = 0.0133333333333; 
                    }
                } else {
                    classes[0] = 0.0; 
                    classes[1] = 0.0; 
                    classes[2] = 0.286666666667; 
                }
            }
        }
        return classes;
    });
    
    forest.push(function(features) {
        var classes = new Array(3);
        if (features[2] <= 5.14999961853) {
            if (features[2] <= 2.45000004768) {
                classes[0] = 8.3290724464e-05; 
                classes[1] = 0.0; 
                classes[2] = 0.0; 
            } else {
                if (features[3] <= 1.75) {
                    if (features[0] <= 4.94999980927) {
                        classes[0] = 0.0; 
                        classes[1] = 1.66581448928e-06; 
                        classes[2] = 1.66581448928e-06; 
                    } else {
                        classes[0] = 0.0; 
                        classes[1] = 0.499954190102; 
                        classes[2] = 3.33162897856e-06; 
                    }
                } else {
                    if (features[1] <= 3.15000009537) {
                        classes[0] = 0.0; 
                        classes[1] = 0.0; 
                        classes[2] = 1.99897738714e-05; 
                    } else {
                        classes[0] = 0.0; 
                        classes[1] = 1.66581448928e-06; 
                        classes[2] = 1.66581448928e-06; 
                    }
                }
            }
        } else {
            classes[0] = 0.0; 
            classes[1] = 0.0; 
            classes[2] = 0.499932534513; 
        }
        return classes;
    });
    
    forest.push(function(features) {
        var classes = new Array(3);
        if (features[3] <= 1.54999995232) {
            if (features[2] <= 4.94999980927) {
                if (features[3] <= 0.800000011921) {
                    classes[0] = 2.67881771865e-08; 
                    classes[1] = 0.0; 
                    classes[2] = 0.0; 
                } else {
                    classes[0] = 0.0; 
                    classes[1] = 0.000184731094993; 
                    classes[2] = 0.0; 
                }
            } else {
                classes[0] = 0.0; 
                classes[1] = 0.0; 
                classes[2] = 0.499696643102; 
            }
        } else {
            if (features[2] <= 5.14999961853) {
                if (features[3] <= 1.84999990463) {
                    if (features[0] <= 5.40000009537) {
                        classes[0] = 0.0; 
                        classes[1] = 0.0; 
                        classes[2] = 0.000111473015249; 
                    } else {
                        classes[0] = 0.0; 
                        classes[1] = 0.499734857502; 
                        classes[2] = 2.67881771865e-09; 
                    }
                } else {
                    classes[0] = 0.0; 
                    classes[1] = 0.0; 
                    classes[2] = 0.000111476765594; 
                }
            } else {
                classes[0] = 0.0; 
                classes[1] = 0.0; 
                classes[2] = 0.000160789052777; 
            }
        }
        return classes;
    });
    
    forest.push(function(features) {
        var classes = new Array(3);
        if (features[3] <= 1.75) {
            if (features[3] <= 1.54999995232) {
                if (features[2] <= 4.94999980927) {
                    if (features[3] <= 0.800000011921) {
                        classes[0] = 9.25765397376e-11; 
                        classes[1] = 0.0; 
                        classes[2] = 0.0; 
                    } else {
                        classes[0] = 0.0; 
                        classes[1] = 6.38407213652e-07; 
                        classes[2] = 0.0; 
                    }
                } else {
                    classes[0] = 0.0; 
                    classes[1] = 0.0; 
                    classes[2] = 0.00172688816469; 
                }
            } else {
                if (features[0] <= 6.94999980927) {
                    if (features[1] <= 2.59999990463) {
                        classes[0] = 0.0; 
                        classes[1] = 0.0; 
                        classes[2] = 3.85236589785e-07; 
                    } else {
                        classes[0] = 0.0; 
                        classes[1] = 0.499024234255; 
                        classes[2] = 0.0; 
                    }
                } else {
                    classes[0] = 0.0; 
                    classes[1] = 0.0; 
                    classes[2] = 5.55607306084e-07; 
                }
            }
        } else {
            if (features[1] <= 3.15000009537) {
                classes[0] = 0.0; 
                classes[1] = 0.0; 
                classes[2] = 0.499135573641; 
            } else {
                if (features[2] <= 4.94999980927) {
                    classes[0] = 0.0; 
                    classes[1] = 0.000111339336392; 
                    classes[2] = 0.0; 
                } else {
                    classes[0] = 0.0; 
                    classes[1] = 0.0; 
                    classes[2] = 3.85258808154e-07; 
                }
            }
        }
        return classes;
    });
    
    this.predict = function(features) {
        var n_estimators = forest.length;
        var preds = new Array(n_estimators);
        var n_classes = 3;
        var classes = new Array(n_classes).fill(0.);
        var normalizer, sum, idx, val;
        var i, j;
    
        for (i = 0; i < n_estimators; i++) {
            preds[i] = forest[i](features);
        }
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
                if (preds[i][j] <= 2.2204460492503131e-16) {
                    preds[i][j] = 2.2204460492503131e-16;
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
        for (i = 0; i < n_estimators; i++) {
            for (j = 0; j < n_classes; j++) {
                classes[j] += preds[i][j];
            }
        }
    
        return findMax(classes);
    };
};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 === 4) {

        // Features:
        var features = process.argv.slice(2);

        // Prediction:
        var clf = new AdaBoostClassifier();
        var prediction = clf.predict(features);
        console.log(prediction);

    }
}
"""
