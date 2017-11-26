# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

porter = Porter(clf, language='js')
output = porter.export(embed_data=True)
print(output)

"""
var DecisionTreeClassifier = function() {

    var findMax = function(nums) {
        var index = 0;
        for (var i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    };

    this.predict = function(features) {
        var classes = new Array(3);
            
        if (features[2] <= 2.45000004768) {
            classes[0] = 50; 
            classes[1] = 0; 
            classes[2] = 0; 
        } else {
            if (features[3] <= 1.75) {
                if (features[2] <= 4.94999980927) {
                    if (features[3] <= 1.65000009537) {
                        classes[0] = 0; 
                        classes[1] = 47; 
                        classes[2] = 0; 
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 1; 
                    }
                } else {
                    if (features[3] <= 1.54999995232) {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 3; 
                    } else {
                        if (features[0] <= 6.94999980927) {
                            classes[0] = 0; 
                            classes[1] = 2; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 1; 
                        }
                    }
                }
            } else {
                if (features[2] <= 4.85000038147) {
                    if (features[1] <= 3.09999990463) {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 2; 
                    } else {
                        classes[0] = 0; 
                        classes[1] = 1; 
                        classes[2] = 0; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 43; 
                }
            }
        }
    
        return findMax(classes);
    };

};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 === 4) {

        // Features:
        var features = process.argv.slice(2);

        // Prediction:
        var clf = new DecisionTreeClassifier();
        var prediction = clf.predict(features);
        console.log(prediction);

    }
}
"""
