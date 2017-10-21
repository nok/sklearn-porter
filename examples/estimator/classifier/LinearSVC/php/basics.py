# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

porter = Porter(clf, language='php')
output = porter.export()
print(output)

"""
<?php

class LinearSVC {

    public function __construct($coefficients, $intercepts) {
        $this->coefficients = $coefficients;
        $this->intercepts = $intercepts;
    }

    public function predict($features) {
        $classIdx = -1;
        $classVal = null;
        for ($i = 0, $il = count($this->intercepts); $i < $il; $i++) {
            $prob = 0.;
            for ($j = 0, $jl = count($this->coefficients[0]); $j < $jl; $j++) {
                $prob += $this->coefficients[$i][$j] * $features[$j];
            }
            if (is_null($classVal) || $prob + $this->intercepts[$i] > $classVal) {
                $classVal = $prob + $this->intercepts[$i];
                $classIdx = $i;
            }
        }
        return $classIdx;
    }

}

if ($argc > 1) {

    // Features:
    array_shift($argv);
    $features = $argv;

    // Parameters:
    $coefficients = [[0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858], [0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452], [-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917]];
    $intercepts = [0.10956266406702335, 1.6636707776739579, -1.7096109416521363];

    // Prediction:
    $clf = new LinearSVC($coefficients, $intercepts);
    $prediction = $clf->predict($features);
    fwrite(STDOUT, $prediction);

}
"""
