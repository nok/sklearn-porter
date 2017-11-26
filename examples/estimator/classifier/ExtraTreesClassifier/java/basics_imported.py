# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = ExtraTreesClassifier(n_estimators=15, max_depth=None,
                           min_samples_split=2, random_state=0)
clf.fit(X, y)

porter = Porter(clf)
output = porter.export(export_data=True)
print(output)

"""
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.List;
import java.util.Scanner;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;


class ExtraTreesClassifier {

    private class Tree {
        private int[] childrenLeft;
        private int[] childrenRight;
        private double[] thresholds;
        private int[] indices;
        private double[][] classes;

        private int predict (double[] features, int node) {
            if (this.thresholds[node] != -2) {
                if (features[this.indices[node]] <= this.thresholds[node]) {
                    return this.predict(features, this.childrenLeft[node]);
                } else {
                    return this.predict(features, this.childrenRight[node]);
                }
            }
            return ExtraTreesClassifier.findMax(this.classes[node]);
        }
        private int predict (double[] features) {
            return this.predict(features, 0);
        }
    }

    private List<Tree> forest;
    private int nClasses;
    private int nEstimators;

    public ExtraTreesClassifier (String file) throws FileNotFoundException {
        String jsonStr = new Scanner(new File(file)).useDelimiter("\\Z").next();
        Gson gson = new Gson();
        Type listType = new TypeToken<List<Tree>>(){}.getType();
        this.forest = gson.fromJson(jsonStr, listType);
        this.nEstimators = this.forest.size();
        this.nClasses = this.forest.get(0).classes[0].length;
    }

    private static int findMax(double[] nums) {
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    }

    public int predict(double[] features) {
        double[] classes = new double[this.nClasses];
        for (int i = 0; i < this.nEstimators; i++) {
            classes[this.forest.get(i).predict(features, 0)]++;
        }
        return ExtraTreesClassifier.findMax(classes);
    }

    public static void main(String[] args) throws IOException {
        if (args.length > 0 && args[0].endsWith(".json")) {

            // Features:
            double[] features = new double[args.length-1];
            for (int i = 1, l = args.length; i < l; i++) {
                features[i-1] = Double.parseDouble(args[i]);
            }

            // Parameters:
            String modelData = args[0];

            // Estimator:
            ExtraTreesClassifier clf = new ExtraTreesClassifier(modelData);

            // Prediction:
            int prediction = clf.predict(features);
            System.out.println(prediction);

        }
    }
}
"""
