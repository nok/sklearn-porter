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
output = porter.export()
print(output)

"""
class DecisionTreeClassifier {

    private int[] lChilds;
    private int[] rChilds;
    private double[] thresholds;
    private int[] indices;
    private int[][] classes;

    public DecisionTreeClassifier(int[] lChilds, int[] rChilds, double[] thresholds, int[] indices, int[][] classes) {
        this.lChilds = lChilds;
        this.rChilds = rChilds;
        this.thresholds = thresholds;
        this.indices = indices;
        this.classes = classes;
    }

    public int predict(double[] features) {
        return this.predict(features, 0);
    }

    public int predict(double[] features, int node) {
        if (this.thresholds[node] != -2) {
            if (features[this.indices[node]] <= this.thresholds[node]) {
                return predict(features, this.lChilds[node]);
            } else {
                return predict(features, this.rChilds[node]);
            }
        }
        return findMax(this.classes[node]);
    }

    private int findMax(int[] nums) {
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    }

    public static void main(String[] args) {
        if (args.length == 4) {

            // Features:
            double[] features = new double[args.length];
            for (int i = 0, l = args.length; i < l; i++) {
                features[i] = Double.parseDouble(args[i]);
            }

            // Parameters:
            int[] lChilds = {1, -1, 3, 4, 5, -1, -1, 8, -1, 10, -1, -1, 13, 14, -1, -1, -1};
            int[] rChilds = {2, -1, 12, 7, 6, -1, -1, 9, -1, 11, -1, -1, 16, 15, -1, -1, -1};
            double[] thresholds = {2.45000004768, -2.0, 1.75, 4.94999980927, 1.65000009537, -2.0, -2.0, 1.54999995232, -2.0, 6.94999980927, -2.0, -2.0, 4.85000038147, 3.09999990463, -2.0, -2.0, -2.0};
            int[] indices = {2, 2, 3, 2, 3, 2, 2, 3, 2, 0, 2, 2, 2, 1, 2, 2, 2};
            int[][] classes = {{50, 50, 50}, {50, 0, 0}, {0, 50, 50}, {0, 49, 5}, {0, 47, 1}, {0, 47, 0}, {0, 0, 1}, {0, 2, 4}, {0, 0, 3}, {0, 2, 1}, {0, 2, 0}, {0, 0, 1}, {0, 1, 45}, {0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 0, 43}};

            // Prediction:
            DecisionTreeClassifier clf = new DecisionTreeClassifier(lChilds, rChilds, thresholds, indices, classes);
            int estimation = clf.predict(features);
            System.out.println(estimation);

        }
    }
}
"""
