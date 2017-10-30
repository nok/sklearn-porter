# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

porter = Porter(clf, language='c')
output = porter.export()
print(output)

"""
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N_FEATURES 4
#define N_CLASSES 3

int lChilds[17] = {1, -1, 3, 4, 5, -1, -1, 8, -1, 10, -1, -1, 13, 14, -1, -1, -1};
int rChilds[17] = {2, -1, 12, 7, 6, -1, -1, 9, -1, 11, -1, -1, 16, 15, -1, -1, -1};
double thresholds[17] = {2.45000004768, -2.0, 1.75, 4.94999980927, 1.65000009537, -2.0, -2.0, 1.54999995232, -2.0, 6.94999980927, -2.0, -2.0, 4.85000038147, 3.09999990463, -2.0, -2.0, -2.0};
int indices[17] = {2, 2, 3, 2, 3, 2, 2, 3, 2, 0, 2, 2, 2, 1, 2, 2, 2};
int classes[17][3] = {{50, 50, 50}, {50, 0, 0}, {0, 50, 50}, {0, 49, 5}, {0, 47, 1}, {0, 47, 0}, {0, 0, 1}, {0, 2, 4}, {0, 0, 3}, {0, 2, 1}, {0, 2, 0}, {0, 0, 1}, {0, 1, 45}, {0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 0, 43}};

int findMax(int nums[N_CLASSES]) {
    int index = 0;
    for (int i = 0; i < N_CLASSES; i++) {
        index = nums[i] > nums[index] ? i : index;
    }
    return index;
}

int predict(double features[N_FEATURES], int node) {
    if (thresholds[node] != -2) {
        if (features[indices[node]] <= thresholds[node]) {
            return predict(features, lChilds[node]);
        } else {
            return predict(features, rChilds[node]);
        }
    }
    return findMax(classes[node]);
}

int main(int argc, const char * argv[]) {

    /* Features: */
    double features[argc-1];
    int i;
    for (i = 1; i < argc; i++) {
        features[i-1] = atof(argv[i]);
    }

    /* Prediction: */
    printf("%d", predict(features, 0));
    return 0;

}
"""
