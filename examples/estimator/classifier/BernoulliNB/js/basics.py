# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.naive_bayes import BernoulliNB
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = BernoulliNB()
clf.fit(X, y)

porter = Porter(clf, language='js')
output = porter.export()
print(output)

"""
var BernoulliNB = function(priors, negProbs, delProbs) {

    this.priors = priors;
    this.negProbs = negProbs;
    this.delProbs = delProbs;

    this.predict = function(features) {
        var jll = new Array(3);
    
        for (var i = 0; i < 3; i++) {
            var sum = 0.;
            for (var j = 0; j < 4; j++) {
                sum += features[i] * this.delProbs[j][i];
            }
            jll[i] = sum;
        }
        for (var i = 0; i < 3; i++) {
            var sum = 0.;
            for (var j = 0; j < 4; j++) {
                sum += this.negProbs[i][j];
            }
            jll[i] += this.priors[i] + sum;
        }
        var classIdx = 0;
    
        for (var i = 0, l = 3; i < l; i++) {
            classIdx = jll[i] > jll[classIdx] ? i : classIdx;
        }
        return classIdx;
    };

};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 === 4) {

        // Features:
        var features = process.argv.slice(2);

        // Parameters:
        var priors = [-1.0986122886681096, -1.0986122886681096, -1.0986122886681096];
        var negProbs = [[-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138], [-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138], [-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138]];
        var delProbs = [[3.931825632724312, 3.931825632724312, 3.931825632724312], [3.931825632724312, 3.931825632724312, 3.931825632724312], [3.931825632724312, 3.931825632724312, 3.931825632724312], [3.931825632724312, 3.931825632724312, 3.931825632724312]];

        // Estimator:
        var clf = new BernoulliNB(priors, negProbs, delProbs);
        var prediction = clf.predict(features);
        console.log(prediction);

    }
}
"""
