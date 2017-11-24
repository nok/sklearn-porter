# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = KNeighborsClassifier(algorithm='brute',
                           n_neighbors=3,
                           weights='uniform')
clf.fit(X, y)

porter = Porter(clf)
output = porter.export(export_data=True)
print(output)

"""
import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import com.google.gson.Gson;


class KNeighborsClassifier {

    private class Classifier {
        private int kNeighbors;
        private int nClasses;
        private double power;
        private double[][] X;
        private int[] y;
    }

    private class Sample {
        Integer y;
        Double dist;
        private Sample(int y, double distance) {
            this.y = y;
            this.dist = distance;
        }
    }

    private Classifier clf;
    private int nTemplates;

    public KNeighborsClassifier(String file) throws FileNotFoundException {
        String jsonStr = new Scanner(new File(file)).useDelimiter("\\Z").next();
        this.clf = new Gson().fromJson(jsonStr, Classifier.class);
        this.nTemplates = this.clf.y.length;
    }

    private static double compute(double[] temp, double[] cand, double q) {
        double dist = 0.;
        double diff;
        for (int i = 0, l = temp.length; i < l; i++) {
            diff = Math.abs(temp[i] - cand[i]);
            if (q==1) {
                dist += diff;
            } else if (q==2) {
                dist += diff*diff;
            } else if (q==Double.POSITIVE_INFINITY) {
                if (diff > dist) {
                    dist = diff;
                }
            } else {
                dist += Math.pow(diff, q);
            }
        }
        if (q==1 || q==Double.POSITIVE_INFINITY) {
            return dist;
        } else if (q==2) {
            return Math.sqrt(dist);
        } else {
            return Math.pow(dist, 1. / q);
        }
    }

    public int predict(double[] features) {
        int classIdx = 0;
        if (this.clf.kNeighbors == 1) {
            double minDist = Double.POSITIVE_INFINITY;
            double curDist;
            for (int i = 0; i < this.nTemplates; i++) {
                curDist = KNeighborsClassifier.compute(this.clf.X[i],
                        features, this.clf.power);
                if (curDist <= minDist) {
                    minDist = curDist;
                    classIdx = this.clf.y[i];
                }
            }
        } else {
            int[] classes = new int[this.clf.nClasses];
            ArrayList<Sample> dists = new ArrayList<Sample>();
            for (int i = 0; i < this.nTemplates; i++) {
                double dist = KNeighborsClassifier.compute(
                        this.clf.X[i], features, this.clf.power);
                dists.add(new Sample(this.clf.y[i], dist));
            }
            Collections.sort(dists, new Comparator<Sample>() {
                @Override
                public int compare(Sample n1, Sample n2) {
                    return n1.dist.compareTo(n2.dist);
                }
            });
            for (Sample neighbor : dists.subList(0, this.clf.kNeighbors)) {
                classes[neighbor.y]++;
            }
            for (int i = 0; i < this.clf.nClasses; i++) {
                classIdx = classes[i] > classes[classIdx] ? i : classIdx;
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
            KNeighborsClassifier clf = new KNeighborsClassifier(modelData);

            // Prediction:
            int prediction = clf.predict(features);
            System.out.println(prediction);

        }
    }
}
"""
