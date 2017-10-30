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
output = porter.export()
print(output)

"""
var DecisionTreeClassifier = function(lChilds, rChilds, thresholds, indices, classes) {

    this.lChilds = lChilds;
    this.rChilds = rChilds;
    this.thresholds = thresholds;
    this.indices = indices;
    this.classes = classes;

    this.findMax = function(nums) {
        var index = 0;
        for (var i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    };

    this.predict = function(features, node) {
        node = (typeof node !== 'undefined') ? node : 0;
        if (this.thresholds[node] != -2) {
            if (features[this.indices[node]] <= this.thresholds[node]) {
                return this.predict(features, this.lChilds[node]);
            } else {
                return this.predict(features, this.rChilds[node]);
            }
        }
        return this.findMax(this.classes[node]);
    };

};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 === 4) {

        // Features:
        var features = process.argv.slice(2);

        // Parameters:
        var lChilds = [1, -1, 3, 4, 5, -1, -1, 8, -1, 10, -1, -1, 13, 14, -1, -1, -1];
        var rChilds = [2, -1, 12, 7, 6, -1, -1, 9, -1, 11, -1, -1, 16, 15, -1, -1, -1];
        var thresholds = [2.45000004768, -2.0, 1.75, 4.94999980927, 1.65000009537, -2.0, -2.0, 1.54999995232, -2.0, 5.44999980927, -2.0, -2.0, 4.85000038147, 3.09999990463, -2.0, -2.0, -2.0];
        var indices = [2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2];
        var classes = [[50, 50, 50], [50, 0, 0], [0, 50, 50], [0, 49, 5], [0, 47, 1], [0, 47, 0], [0, 0, 1], [0, 2, 4], [0, 0, 3], [0, 2, 1], [0, 2, 0], [0, 0, 1], [0, 1, 45], [0, 1, 2], [0, 0, 2], [0, 1, 0], [0, 0, 43]];

        // Prediction:
        var clf = new DecisionTreeClassifier(lChilds, rChilds, thresholds, indices, classes);
        var prediction = clf.predict(features);
        console.log(prediction);

    }
}
"""
