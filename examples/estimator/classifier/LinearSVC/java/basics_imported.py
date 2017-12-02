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
output = porter.export(export_data=True)
print(output)

"""
import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import com.google.gson.Gson;


class LinearSVC {

    private class Classifier {
        private double[][] coefficients;
        private double[] intercepts;
    }

    private Classifier clf;

    public LinearSVC(String file) throws FileNotFoundException {
        String jsonStr = new Scanner(new File(file)).useDelimiter("\\Z").next();
        this.clf = new Gson().fromJson(jsonStr, Classifier.class);
    }

    public int predict(double[] features) {
        int classIdx = 0;
        double classVal = Double.NEGATIVE_INFINITY;
        for (int i = 0, il = this.clf.intercepts.length; i < il; i++) {
            double prob = 0.;
            for (int j = 0, jl = this.clf.coefficients[0].length; j < jl; j++) {
                prob += this.clf.coefficients[i][j] * features[j];
            }
            if (prob + this.clf.intercepts[i] > classVal) {
                classVal = prob + this.clf.intercepts[i];
                classIdx = i;
            }
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
            LinearSVC clf = new LinearSVC(modelData);

            // Prediction:
            int prediction = clf.predict(features);
            System.out.println(prediction);

        }
    }
}
"""
