# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

porter = Porter(clf, language='php')
output = porter.export()
print(output)

"""
<?php

class DecisionTreeClassifier {

    public function __construct($lChilds, $rChilds, $thresholds, $indices, $classes) {
        $this->lChilds = $lChilds;
        $this->rChilds = $rChilds;
        $this->thresholds = $thresholds;
        $this->indices = $indices;
        $this->classes = $classes;
    }

    private function findMax($nums) {
        $index = 0;
        for ($i = 0; $i < count($nums); $i++) {
            $index = $nums[$i] > $nums[$index] ? $i : $index;
        }
        return $index;
    }

    public function predict($features) {
        $node = (func_num_args() > 1) ? func_get_arg(1) : 0;
        if ($this->thresholds[$node] != -2) {
            if ($features[$this->indices[$node]] <= $this->thresholds[$node]) {
                return $this->predict($features, $this->lChilds[$node]);
            } else {
                return $this->predict($features, $this->rChilds[$node]);
            }
        }
        return $this->findMax($this->classes[$node]);
    }

}

if ($argc > 1) {

    // Features:
    array_shift($argv);
    $features = $argv;

    // Parameters:
    $lChilds = [1, -1, 3, 4, 5, -1, -1, 8, -1, 10, -1, -1, 13, 14, -1, -1, -1];
    $rChilds = [2, -1, 12, 7, 6, -1, -1, 9, -1, 11, -1, -1, 16, 15, -1, -1, -1];
    $thresholds = [2.45000004768, -2.0, 1.75, 4.94999980927, 1.65000009537, -2.0, -2.0, 1.54999995232, -2.0, 5.44999980927, -2.0, -2.0, 4.85000038147, 5.94999980927, -2.0, -2.0, -2.0];
    $indices = [2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 0, 2, 2, 2];
    $classes = [[50, 50, 50], [50, 0, 0], [0, 50, 50], [0, 49, 5], [0, 47, 1], [0, 47, 0], [0, 0, 1], [0, 2, 4], [0, 0, 3], [0, 2, 1], [0, 2, 0], [0, 0, 1], [0, 1, 45], [0, 1, 2], [0, 1, 0], [0, 0, 2], [0, 0, 43]];

    // Prediction:
    $clf = new DecisionTreeClassifier($lChilds, $rChilds, $thresholds, $indices, $classes);
    $prediction = $clf->predict($features);
    fwrite(STDOUT, $prediction);

}
"""
