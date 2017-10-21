# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

porter = Porter(clf, language='c')
output = porter.export()
print(output)

"""
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double coefficients[3][4] = {{0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858}, {0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452}, {-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917}};
double intercepts[3] = {0.10956266406702335, 1.6636707776739579, -1.7096109416521363};

int predict (float features[]) {
    double class_val = -INFINITY;
    int class_idx = -1;
    int i, il, j, jl;
    for (i = 0, il = sizeof(coefficients) / sizeof (coefficients[0]); i < il; i++) {
        double prob = 0.;
        for (j = 0, jl = sizeof(coefficients[0]) / sizeof (coefficients[0][0]); j < jl; j++) {
            prob += coefficients[i][j] * features[j];
        }
        if (prob + intercepts[i] > class_val) {
            class_val = prob + intercepts[i];
            class_idx = i;
        }
    }
    return class_idx;
}

int main(int argc, const char * argv[]) {

    /* Features: */
    float features[argc-1];
    int i;
    for (i = 1; i < argc; i++) {
        features[i-1] = atof(argv[i]);
    }

    /* Prediction: */
    printf("%d", predict(features));
    return 0;

}
"""
