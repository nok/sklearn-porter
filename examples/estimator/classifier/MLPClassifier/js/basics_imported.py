# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn_porter import Porter


iris_data = load_iris()
X, y = iris_data.data, iris_data.target

X = shuffle(X, random_state=0)
y = shuffle(y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=5)

clf = MLPClassifier(
    activation='relu', hidden_layer_sizes=50, max_iter=500, alpha=1e-4,
    solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)

clf.fit(X_train, y_train)

porter = Porter(clf, language='js')
output = porter.export(export_data=True)
print(output)

"""
if (typeof XMLHttpRequest === 'undefined') {
    var XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest;
}

var MLPClassifier = function(jsonFile) {
    this.mdl = undefined;

    var promise = new Promise(function(resolve, reject) {
        var httpRequest = new XMLHttpRequest();
        httpRequest.onreadystatechange = function() {
            if (httpRequest.readyState === 4) {
                if (httpRequest.status === 200) {
                    resolve(JSON.parse(httpRequest.responseText));
                } else {
                    reject(new Error(httpRequest.statusText));
                }
            }
        };
        httpRequest.open('GET', jsonFile, true);
        httpRequest.send();
    });

    // Return max index:
    var maxi = function(nums) {
        var index = 0;
        for (var i=0, l=nums.length; i < l; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    };

    // Compute the activation function:
    var compute = function(activation, v) {
        switch (activation) {
            case 'LOGISTIC':
                for (var i = 0, l = v.length; i < l; i++) {
                    v[i] = 1. / (1. + Math.exp(-v[i]));
                }
                break;
            case 'RELU':
                for (var i = 0, l = v.length; i < l; i++) {
                    v[i] = Math.max(0, v[i]);
                }
                break;
            case 'TANH':
                for (var i = 0, l = v.length; i < l; i++) {
                    v[i] = Math.tanh(v[i]);
                }
                break;
            case 'SOFTMAX':
                var max = Number.NEGATIVE_INFINITY;
                for (var i = 0, l = v.length; i < l; i++) {
                    if (v[i] > max) {
                        max = v[i];
                    }
                }
                for (var i = 0, l = v.length; i < l; i++) {
                    v[i] = Math.exp(v[i] - max);
                }
                var sum = 0.0;
                for (var i = 0, l = v.length; i < l; i++) {
                    sum += v[i];
                }
                for (var i = 0, l = v.length; i < l; i++) {
                    v[i] /= sum;
                }
                break;
        }
        return v;
    };

    this.predict = function(neurons) {
        return new Promise(function(resolve, reject) {
            promise.then(function(mdl) {

                // Initialization:
                if (typeof this.mdl === 'undefined') {
                    mdl.hidden_activation = mdl.hidden_activation.toUpperCase();
                    mdl.output_activation = mdl.output_activation.toUpperCase();
                    mdl.network = new Array(mdl.layers.length + 1);
                    for (var i = 0, l = mdl.layers.length; i < l; i++) {
                        mdl.network[i + 1] = new Array(mdl.layers[i]).fill(0.);
                    }
                    this.mdl = mdl;
                }

                // Feed forward:
                this.mdl.network[0] = neurons;
                for (var i = 0; i < this.mdl.network.length - 1; i++) {
                    for (var j = 0; j < this.mdl.network[i + 1].length; j++) {
                        for (var l = 0; l < this.mdl.network[i].length; l++) {
                            this.mdl.network[i + 1][j] += this.mdl.network[i][l] * this.mdl.weights[i][l][j];
                        }
                        this.mdl.network[i + 1][j] += this.mdl.bias[i][j];
                    }
                    if ((i + 1) < (this.mdl.network.length - 1)) {
                        this.mdl.network[i + 1] = compute(this.mdl.hidden_activation, this.mdl.network[i + 1]);
                    }
                }
                this.mdl.network[this.mdl.network.length - 1] = compute(this.mdl.output_activation, this.mdl.network[this.mdl.network.length - 1]);

                // Return result:
                if (this.mdl.network[this.mdl.network.length - 1].length == 1) {
                    if (this.mdl.network[this.mdl.network.length - 1][0] > .5) {
                        resolve(1);
                    }
                    resolve(0);
                } else {
                    resolve(maxi(this.mdl.network[this.mdl.network.length - 1]));
                }
            }, function(error) {
                reject(error);
            });
        });
    };
};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv[2].trim().endsWith('.json')) {

        // Features:
        var features = process.argv.slice(3);

        // Parameters:
        var json = process.argv[2];

        // Estimator:
        var clf = new MLPClassifier(json);

        // Prediction:
        clf.predict(features).then(function(prediction) {
            console.log(prediction);
        });

    }
}
"""
