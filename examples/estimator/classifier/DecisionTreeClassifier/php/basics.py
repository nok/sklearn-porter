# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X, y = iris_data.data, iris_data.target

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

output = Porter(clf, language='php').export()
print(output)

"""
<?php

class Brain {

    public static function predict($atts) {
        if (sizeof($atts) != 4) { return -1; }

        $classes = array_fill(0, 3, 0);

        if ($atts[3] <= 0.80000001192092896) {
            $classes[0] = 50;
            $classes[1] = 0;
            $classes[2] = 0;
        } else {
            if ($atts[3] <= 1.75) {
                if ($atts[2] <= 4.9499998092651367) {
                    if ($atts[3] <= 1.6500000953674316) {
                        $classes[0] = 0;
                        $classes[1] = 47;
                        $classes[2] = 0;
                    } else {
                        $classes[0] = 0;
                        $classes[1] = 0;
                        $classes[2] = 1;
                    }
                } else {
                    if ($atts[3] <= 1.5499999523162842) {
                        $classes[0] = 0;
                        $classes[1] = 0;
                        $classes[2] = 3;
                    } else {
                        if ($atts[2] <= 5.4499998092651367) {
                            $classes[0] = 0;
                            $classes[1] = 2;
                            $classes[2] = 0;
                        } else {
                            $classes[0] = 0;
                            $classes[1] = 0;
                            $classes[2] = 1;
                        }
                    }
                }
            } else {
                if ($atts[2] <= 4.8500003814697266) {
                    if ($atts[0] <= 5.9499998092651367) {
                        $classes[0] = 0;
                        $classes[1] = 1;
                        $classes[2] = 0;
                    } else {
                        $classes[0] = 0;
                        $classes[1] = 0;
                        $classes[2] = 2;
                    }
                } else {
                    $classes[0] = 0;
                    $classes[1] = 0;
                    $classes[2] = 43;
                }
            }
        }

        $class_idx = 0;
        $class_val = $classes[0];

        for ($i = 1; $i < 3; $i++) {
            if ($classes[$i] > $class_val) {
                $class_idx = $i;
                $class_val = $classes[$i];
            }
        }
        return $class_idx;
    }

}

if ($argc > 1) {
    array_shift($argv);
    $prediction = Brain::predict($argv);
    fwrite(STDOUT, $prediction);
}
"""
