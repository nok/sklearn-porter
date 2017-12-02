# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = GaussianNB()
clf.fit(X, y)

porter = Porter(clf)
output = porter.export(export_data=True)
print(output)

"""
import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import com.google.gson.Gson;


class GaussianNB {

    private class Classifier {
        private double[] priors;
        private double[][] sigmas;
        private double[][] thetas;
    }

    private Classifier clf;

    public GaussianNB(String file) throws FileNotFoundException {
        String jsonStr = new Scanner(new File(file)).useDelimiter("\\Z").next();
        this.clf = new Gson().fromJson(jsonStr, Classifier.class);
    }

    public int predict(double[] features) {
        double[] likelihoods = new double[this.clf.sigmas.length];

        for (int i = 0, il = this.clf.sigmas.length; i < il; i++) {
            double sum = 0.;
            for (int j = 0, jl = this.clf.sigmas[0].length; j < jl; j++) {
                sum += Math.log(2. * Math.PI * this.clf.sigmas[i][j]);
            }
            double nij = -0.5 * sum;
            sum = 0.;
            for (int j = 0, jl = this.clf.sigmas[0].length; j < jl; j++) {
                sum += Math.pow(features[j] - this.clf.thetas[i][j], 2.) / this.clf.sigmas[i][j];
            }
            nij -= 0.5 * sum;
            likelihoods[i] = Math.log(this.clf.priors[i]) + nij;
        }

        int classIdx = 0;
        for (int i = 0, l = likelihoods.length; i < l; i++) {
            classIdx = likelihoods[i] > likelihoods[classIdx] ? i : classIdx;
        }
        return classIdx;
    }

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length > 0 && args[0].endsWith(".json")) {

            // Features:
            double[] features = new double[args.length-1];
            for (int i = 1, l = args.length; i < l; i++) {
                features[i - 1] = Double.parseDouble(args[i]);
            }

            // Parameters:
            String modelData = args[0];

            // Estimators:
            GaussianNB clf = new GaussianNB(modelData);

            // Prediction:
            int prediction = clf.predict(features);
            System.out.println(prediction);

        }
    }
}
"""
