# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

porter = Porter(clf, language='go')
output = porter.export()
print(output)

"""
package main

import (
	"os"
	"fmt"
	"strconv"
)

type DecisionTreeClassifier struct {
	lChilds []int
	rChilds []int
	thresholds []float64
	indices []int
	classes [][]int
}

func (dtc DecisionTreeClassifier) predict_(features []float64, node int) int {
    if dtc.thresholds[node] != -2 {
        if features[dtc.indices[node]] <= dtc.thresholds[node] {
            return dtc.predict_(features, dtc.lChilds[node])
        } else {
            return dtc.predict_(features, dtc.rChilds[node])
        }
    }
    var index int = 0
	for i := 0; i < len(dtc.classes[node]); i++ {
	    if dtc.classes[node][i] > dtc.classes[node][index] {
	        index = i
	    }
	}
	return index
}

func (dtc DecisionTreeClassifier) predict(features []float64) int {
    return dtc.predict_(features, 0)
}

func main() {

	// Features:
	var features []float64
	for _, arg := range os.Args[1:] {
		if n, err := strconv.ParseFloat(arg, 64); err == nil {
			features = append(features, n)
		}
	}

    // Parameters:
    lChilds := []int {1, -1, 3, 4, 5, -1, -1, 8, -1, 10, -1, -1, 13, 14, -1, -1, -1}
    rChilds := []int {2, -1, 12, 7, 6, -1, -1, 9, -1, 11, -1, -1, 16, 15, -1, -1, -1}
    thresholds := []float64 {2.45000004768, -2.0, 1.75, 4.94999980927, 1.65000009537, -2.0, -2.0, 1.54999995232, -2.0, 5.44999980927, -2.0, -2.0, 4.85000038147, 5.94999980927, -2.0, -2.0, -2.0}
    indices := []int {2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 0, 2, 2, 2}
    classes := [][]int {{50, 50, 50}, {50, 0, 0}, {0, 50, 50}, {0, 49, 5}, {0, 47, 1}, {0, 47, 0}, {0, 0, 1}, {0, 2, 4}, {0, 0, 3}, {0, 2, 1}, {0, 2, 0}, {0, 0, 1}, {0, 1, 45}, {0, 1, 2}, {0, 1, 0}, {0, 0, 2}, {0, 0, 43}}

	// Prediction:
	clf := DecisionTreeClassifier{lChilds, rChilds, thresholds, indices, classes}
	estimation := clf.predict(features)
	fmt.Printf("%d\n", estimation)

}
"""
