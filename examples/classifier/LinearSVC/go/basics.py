from sklearn import svm
from sklearn.datasets import load_iris

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

# Cheese!

model = Porter(language='go').port(clf)
print(model)

"""
package main

import (
	"fmt"
	"math"
)

func predict(atts []float64) int {

		coefs := [][]float64{{0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858}, {0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452}, {-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917}}
		inters := []float64{0.10956266406702335, 1.6636707776739579, -1.7096109416521363}

		if len(atts) != len(coefs[0]) {
			return -1
		}

		classIdx := -1
		classVal := math.Inf(-1)
		outerCount, innerCount := len(coefs), len(coefs[0])
		for i := 0; i < outerCount; i++ {
			var prob float64
			for j := 0; j < innerCount; j++ {
				prob = prob + coefs[i][j]*atts[j]
			}
			if prob+inters[i] > classVal {
				classVal = prob + inters[i]
				classIdx = i
			}
		}
		return classIdx
	}

func main() {
	// atts := []float64{ /* values */ }
	// classIdx := predict(atts)
	// fmt.Println(classIdx)
}
"""
