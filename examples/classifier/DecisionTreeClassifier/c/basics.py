# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris

from sklearn_porter import Porter


iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Cheese!

result = Porter(clf, language='c').export()
print(result)

"""
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int predict(float atts[3]) {

    int classes[3];

    if (atts[2] <= 2.4500000476837158) {
        classes[0] = 50;
        classes[1] = 0;
        classes[2] = 0;
    } else {
        if (atts[3] <= 1.75) {
            if (atts[2] <= 4.9499998092651367) {
                if (atts[3] <= 1.6500000953674316) {
                    classes[0] = 0;
                    classes[1] = 47;
                    classes[2] = 0;
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 1;
                }
            } else {
                if (atts[3] <= 1.5499999523162842) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 3;
                } else {
                    if (atts[0] <= 6.9499998092651367) {
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
            if (atts[2] <= 4.8500003814697266) {
                if (atts[1] <= 3.0999999046325684) {
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

    int class_val = -1;
    int class_idx = -1;
    int i;
    for (i = 0; i < 3; i++) {
        if (classes[i] > class_val) {
            class_idx = i;
            class_val = classes[i];
        }
    }
    return class_idx;
}

int main(int argc, const char * argv[]) {
    float atts[argc-1];
    int i;
    for (i = 1; i < argc; i++) {
        atts[i-1] = atof(argv[i]);
    }
    printf("%d", predict(atts));
    return 0;
}
"""