# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

porter = Porter(clf)
output = porter.export(export_data=True)
print(output)

"""
import com.google.gson.Gson;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;


class DecisionTreeClassifier {

    private class Classifier {
        private int[] leftChilds;
        private int[] rightChilds;
        private double[] thresholds;
        private int[] indices;
        private int[][] classes;
    }
    private Classifier clf;

    public DecisionTreeClassifier(String file) throws FileNotFoundException {
        String jsonStr = new Scanner(new File(file)).useDelimiter("\\Z").next();
        this.clf = new Gson().fromJson(jsonStr, Classifier.class);
    }

    public int predict(double[] features, int node) {
        if (this.clf.thresholds[node] != -2) {
            if (features[this.clf.indices[node]] <= this.clf.thresholds[node]) {
                return predict(features, this.clf.leftChilds[node]);
            } else {
                return predict(features, this.clf.rightChilds[node]);
            }
        }
        return findMax(this.clf.classes[node]);
    }
    public int predict(double[] features) {
        return this.predict(features, 0);
    }

    private int findMax(int[] nums) {
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
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
            DecisionTreeClassifier clf = new DecisionTreeClassifier(modelData);

            // Prediction:
            int prediction = clf.predict(features);
            System.out.println(prediction);

        }
    }
}
"""