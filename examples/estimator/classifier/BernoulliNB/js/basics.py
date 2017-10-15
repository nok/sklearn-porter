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
    
        var highestLikeli = -1;
        var classIndex = -1;
        for (var i = 0; i < 3; i++) {
            if (jll[i] > highestLikeli) {
                highestLikeli = jll[i];
                classIndex = i;
            }
        }
        return classIndex;
    };

};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 === 4) {

        // Parameters:
        var priors = [-1.0986122886681096, -1.0986122886681096, -1.0986122886681096];
        var negProbs = [[-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138], [-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138], [-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138]];
        var delProbs = [[3.931825632724312, 3.931825632724312, 3.931825632724312], [3.931825632724312, 3.931825632724312, 3.931825632724312], [3.931825632724312, 3.931825632724312, 3.931825632724312], [3.931825632724312, 3.931825632724312, 3.931825632724312]];

        // Estimator:
        const brain = new BernoulliNB(priors, negProbs, delProbs);
        var features = process.argv.slice(2);
        var prediction = brain.predict(features);
        console.log(prediction);

    }
}
"""
