# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

porter = Porter(clf)
output = porter.export()
print(output)

"""
class LinearSVC {

    private double[][] coefficients;
    private double[] intercepts;
    
    public LinearSVC(double[][] coefficients, double[] intercepts) {
        this.coefficients = coefficients;
        this.intercepts = intercepts;
    }

    public int predict(double[] features) {
        int classIdx = 0;
        double classVal = Double.NEGATIVE_INFINITY;
        for (int i = 0, il = this.intercepts.length; i < il; i++) {
            double prob = 0.;
            for (int j = 0, jl = this.coefficients[0].length; j < jl; j++) {
                prob += this.coefficients[i][j] * features[j];
            }
            if (prob + this.intercepts[i] > classVal) {
                classVal = prob + this.intercepts[i];
                classIdx = i;
            }
        }
        return classIdx;
    }

    public static void main(String[] args) {
        if (args.length == 4) {

            // Features:
            double[] features = new double[args.length];
            for (int i = 0, l = args.length; i < l; i++) {
                features[i] = Double.parseDouble(args[i]);
            }

            // Parameters:
            double[][] coefficients = {{0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858}, {0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452}, {-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917}};
            double[] intercepts = {0.10956266406702335, 1.6636707776739579, -1.7096109416521363};

            // Prediction:
            LinearSVC clf = new LinearSVC(coefficients, intercepts);
            int estimation = clf.predict(features);
            System.out.println(estimation);

        }
    }
}
"""
