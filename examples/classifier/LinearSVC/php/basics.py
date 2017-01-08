# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris

from sklearn_porter import Porter


X, y = load_iris(return_X_y=True)
clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

# Cheese!

result = Porter(language='php').port(clf)
print(result)

"""
<?php

class Tmp {

    public static function predict($atts) {
        if (sizeof($atts) != 4) { return -1; }

        $coefs = [[0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858], [0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452], [-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917]];
        $inters = [0.10956266406702335, 1.6636707776739579, -1.7096109416521363];

        $class_idx = -1;
        $class_val = null;
        for ($i = 0; $i < 3; $i++) {
            $prob = 0.;
            for ($j = 0; $j < 4; $j++) {
                $prob += $coefs[$i][$j] * $atts[$j];
            }
            if (is_null($class_val) || $prob + $inters[$i] > $class_val) {
                $class_val = $prob + $inters[$i];
                $class_idx = $i;
            }
        }
        return $class_idx;
    }

}

if ($argc > 1) {
    array_shift($argv);
    $prediction = Tmp::predict($argv);
    fwrite(STDOUT, $prediction);
}

"""