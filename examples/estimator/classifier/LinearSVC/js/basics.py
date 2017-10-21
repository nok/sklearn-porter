# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

porter = Porter(clf, language='js')
output = porter.export()
print(output)

"""
var LinearSVC = function(coefficients, intercepts) {

    this.coefficients = coefficients;
    this.intercepts = intercepts;

    this.predict = function(features) {
        var classIdx = 0,
            classVal = Number.NEGATIVE_INFINITY,
            prob = 0.;
        for (var i = 0, il = this.intercepts.length; i < il; i++) {
            prob = 0.;
            for (var j = 0, jl = this.coefficients[0].length; j < jl; j++) {
                prob += this.coefficients[i][j] * features[j];
            }
            if (prob + this.intercepts[i] > classVal) {
                classVal = prob + this.intercepts[i];
                classIdx = i;
            }
        }
        return classIdx;
    };

};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 === 4) {

        // Features
        var features = process.argv.slice(2);

        // Parameters:
        var coefficients = [[0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858], [0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452], [-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917]];
        var intercepts = [0.10956266406702335, 1.6636707776739579, -1.7096109416521363];

        // Estimators
        var clf = new LinearSVC(coefficients, intercepts);
        var prediction = clf.predict(features);
        console.log(prediction);

    }
}
"""
