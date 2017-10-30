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
output = porter.export(embedded=True)
print(output)

"""
package main

import (
	"os"
	"fmt"
	"strconv"
)

func predict(features []float64) int {
	var classes [3]float64
		
	if features[3] <= 0.800000011921 {
		classes[0] = 50
		classes[1] = 0
		classes[2] = 0
	} else {
		if features[3] <= 1.75 {
			if features[2] <= 4.94999980927 {
				if features[3] <= 1.65000009537 {
					classes[0] = 0
					classes[1] = 47
					classes[2] = 0
				} else {
					classes[0] = 0
					classes[1] = 0
					classes[2] = 1
				}
			} else {
				if features[3] <= 1.54999995232 {
					classes[0] = 0
					classes[1] = 0
					classes[2] = 3
				} else {
					if features[2] <= 5.44999980927 {
						classes[0] = 0
						classes[1] = 2
						classes[2] = 0
					} else {
						classes[0] = 0
						classes[1] = 0
						classes[2] = 1
					}
				}
			}
		} else {
			if features[2] <= 4.85000038147 {
				if features[0] <= 5.94999980927 {
					classes[0] = 0
					classes[1] = 1
					classes[2] = 0
				} else {
					classes[0] = 0
					classes[1] = 0
					classes[2] = 2
				}
			} else {
				classes[0] = 0
				classes[1] = 0
				classes[2] = 43
			}
		}
	}

    var index int = 0
	for i := 0; i < len(classes); i++ {
	    if classes[i] > classes[index] {
	        index = i
	    }
	}
	return index
}

func main() {

	// Features:
	var features []float64
	for _, arg := range os.Args[1:] {
		if n, err := strconv.ParseFloat(arg, 64); err == nil {
			features = append(features, n)
		}
	}

	// Prediction:
	var estimation = predict(features)
	fmt.Printf("%d\n", estimation)

}
"""
