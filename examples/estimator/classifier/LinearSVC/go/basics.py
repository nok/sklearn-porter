# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = svm.LinearSVC(C=1., random_state=0)
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
	"math"
)

type LinearSVC struct {
	coefficients [][]float64
	intercepts []float64
}

func (svc LinearSVC) predict(features []float64) int {
	classIdx := 0
	classVal := math.Inf(-1)
	outerCount, innerCount := len(svc.coefficients), len(svc.coefficients[0])
	for i := 0; i < outerCount; i++ {
		var prob float64
		for j := 0; j < innerCount; j++ {
			prob = prob + svc.coefficients[i][j] * features[j]
		}
		if prob + svc.intercepts[i] > classVal {
			classVal = prob + svc.intercepts[i]
			classIdx = i
		}
	}
	return classIdx
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
	coefficients := [][]float64{{0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858}, {0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452}, {-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917}}
	intercepts := []float64{0.10956266406702335, 1.6636707776739579, -1.7096109416521363}

	// Prediction:
	clf := LinearSVC{coefficients, intercepts}
	estimation := clf.predict(features)
	fmt.Printf("%d\n", estimation)

}
"""
