# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = GaussianNB()
clf.fit(X, y)

porter = Porter(clf, language='js')
output = porter.export()
print(output)

"""
var GaussianNB = function(priors, sigmas, thetas) {

    this.priors = priors;
    this.sigmas = sigmas;
    this.thetas = thetas;

    this.predict = function(features) {
        var likelihoods = new Array(this.sigmas.length);
    
        for (var i = 0, il = this.sigmas.length; i < il; i++) {
            var sum = 0.;
            for (var j = 0, jl = this.sigmas[0].length; j < jl; j++) {
                sum += Math.log(2. * Math.PI * this.sigmas[i][j]);
            }
            var nij = -0.5 * sum;
            sum = 0.;
            for (var j = 0, jl = this.sigmas[0].length; j < jl; j++) {
                sum += Math.pow(features[j] - this.thetas[i][j], 2.) / this.sigmas[i][j];
            }
            nij -= 0.5 * sum;
            likelihoods[i] = Math.log(this.priors[i]) + nij;
        }
    
        var classIdx = 0;
        for (var i = 0, l = likelihoods.length; i < l; i++) {
            classIdx = likelihoods[i] > likelihoods[classIdx] ? i : classIdx;
        }
        return classIdx;
    };

};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 === 4) {

        // Features:
        var features = process.argv.slice(2);

        // Parameters:
        var priors = [0.33333333333333331, 0.33333333333333331, 0.33333333333333331];
        var sigmas = [[0.12176400309242481, 0.14227600309242491, 0.029504003092424898, 0.011264003092424885], [0.26110400309242499, 0.096500003092424902, 0.21640000309242502, 0.038324003092424869], [0.39625600309242481, 0.10192400309242496, 0.29849600309242508, 0.073924003092424875]];
        var thetas = [[5.0059999999999993, 3.4180000000000006, 1.464, 0.24399999999999991], [5.9359999999999999, 2.7700000000000005, 4.2599999999999998, 1.3259999999999998], [6.5879999999999983, 2.9739999999999998, 5.5519999999999996, 2.0259999999999998]];

        // Estimator:
        var clf = new GaussianNB(priors, sigmas, thetas);
        var prediction = clf.predict(features);
        console.log(prediction);

    }
}
"""
