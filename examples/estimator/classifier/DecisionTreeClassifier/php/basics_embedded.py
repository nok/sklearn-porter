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
output = porter.export(embedded=True)
print(output)

"""
<?php

class DecisionTreeClassifier {

    public static function predict($atts) {
        if (sizeof($atts) != 4) { return -1; }
    
        $classes = array_fill(0, 3, 0);
            
        if ($features[3] <= 0.800000011921) {
            $classes[0] = 50; 
            $classes[1] = 0; 
            $classes[2] = 0; 
        } else {
            if ($features[3] <= 1.75) {
                if ($features[2] <= 4.94999980927) {
                    if ($features[3] <= 1.65000009537) {
                        $classes[0] = 0; 
                        $classes[1] = 47; 
                        $classes[2] = 0; 
                    } else {
                        $classes[0] = 0; 
                        $classes[1] = 0; 
                        $classes[2] = 1; 
                    }
                } else {
                    if ($features[3] <= 1.54999995232) {
                        $classes[0] = 0; 
                        $classes[1] = 0; 
                        $classes[2] = 3; 
                    } else {
                        if ($features[0] <= 6.94999980927) {
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
                if ($features[2] <= 4.85000038147) {
                    if ($features[0] <= 5.94999980927) {
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
    $prediction = DecisionTreeClassifier::predict($argv);
    fwrite(STDOUT, $prediction);
}
"""
