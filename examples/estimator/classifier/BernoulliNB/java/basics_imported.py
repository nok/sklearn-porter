# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.naive_bayes import BernoulliNB
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = BernoulliNB()
clf.fit(X, y)

porter = Porter(clf)
output = porter.export(export_data=True)
print(output)

"""
import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import com.google.gson.Gson;


class BernoulliNB {

    private class Classifier {
        private double[] priors;
        private double[][] negProbs;
        private double[][] delProbs;
    }

    private Classifier clf;

    public BernoulliNB(String file) throws FileNotFoundException {
        String jsonStr = new Scanner(new File(file)).useDelimiter("\\Z").next();
        this.clf = new Gson().fromJson(jsonStr, Classifier.class);
    }

    public int predict(double[] features) {
        int nClasses = this.clf.priors.length;
        int nFeatures = this.clf.delProbs.length;

        double[] jll = new double[nClasses];
        for (int i = 0; i < nClasses; i++) {
            double sum = 0.;
            for (int j = 0; j < nFeatures; j++) {
                sum += features[i] * this.clf.delProbs[j][i];
            }
            jll[i] = sum;
        }
        for (int i = 0; i < nClasses; i++) {
            double sum = 0.;
            for (int j = 0; j < nFeatures; j++) {
                sum += this.clf.negProbs[i][j];
            }
            jll[i] += this.clf.priors[i] + sum;
        }

        int classIndex = 0;
        for (int i = 0; i < nClasses; i++) {
            classIndex = jll[i] > jll[classIndex] ? i : classIndex;
        }
        return classIndex;
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
            BernoulliNB clf = new BernoulliNB(modelData);

            // Prediction:
            int prediction = clf.predict(features);
            System.out.println(prediction);

        }
    }
}
"""
