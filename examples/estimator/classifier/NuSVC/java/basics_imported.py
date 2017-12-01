# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = svm.NuSVC(gamma=0.001, kernel='rbf', random_state=0)
clf.fit(X, y)

porter = Porter(clf)
output = porter.export(export_data=True)
print(output)

"""
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import com.google.gson.Gson;


class NuSVC {

    private enum Kernel { LINEAR, POLY, RBF, SIGMOID }
    private class Classifier {
        private int nClasses;
        private int nRows;
        private int[] classes;
        private double[][] vectors;
        private double[][] coefficients;
        private double[] intercepts;
        private int[] weights;
        private String kernel;
        private Kernel kkernel;
        private double gamma;
        private double coef0;
        private double degree;
    }

    private Classifier clf;

    public NuSVC(String file) throws FileNotFoundException {
        String jsonStr = new Scanner(new File(file)).useDelimiter("\\Z").next();
        this.clf = new Gson().fromJson(jsonStr, Classifier.class);
        this.clf.classes = new int[this.clf.nClasses];
        for (int i = 0; i < this.clf.nClasses; i++) {
            this.clf.classes[i] = i;
        }
        this.clf.kkernel = Kernel.valueOf(this.clf.kernel.toUpperCase());
    }

    public int predict(double[] features) {
        double[] kernels = new double[this.clf.vectors.length];
        double kernel;
        switch (this.clf.kkernel) {
            case LINEAR:
                // <x,x'>
                for (int i = 0; i < this.clf.vectors.length; i++) {
                    kernel = 0.;
                    for (int j = 0; j < this.clf.vectors[i].length; j++) {
                        kernel += this.clf.vectors[i][j] * features[j];
                    }
                    kernels[i] = kernel;
                }
                break;
            case POLY:
                // (y<x,x'>+r)^d
                for (int i = 0; i < this.clf.vectors.length; i++) {
                    kernel = 0.;
                    for (int j = 0; j < this.clf.vectors[i].length; j++) {
                        kernel += this.clf.vectors[i][j] * features[j];
                    }
                    kernels[i] = Math.pow((this.clf.gamma * kernel) + this.clf.coef0, this.clf.degree);
                }
                break;
            case RBF:
                // exp(-y|x-x'|^2)
                for (int i = 0; i < this.clf.vectors.length; i++) {
                    kernel = 0.;
                    for (int j = 0; j < this.clf.vectors[i].length; j++) {
                        kernel += Math.pow(this.clf.vectors[i][j] - features[j], 2);
                    }
                    kernels[i] = Math.exp(-this.clf.gamma * kernel);
                }
                break;
            case SIGMOID:
                // tanh(y<x,x'>+r)
                for (int i = 0; i < this.clf.vectors.length; i++) {
                    kernel = 0.;
                    for (int j = 0; j < this.clf.vectors[i].length; j++) {
                        kernel += this.clf.vectors[i][j] * features[j];
                    }
                    kernels[i] = Math.tanh((this.clf.gamma * kernel) + this.clf.coef0);
                }
                break;
        }

        int[] starts = new int[this.clf.nRows];
        for (int i = 0; i < this.clf.nRows; i++) {
            if (i != 0) {
                int start = 0;
                for (int j = 0; j < i; j++) {
                    start += this.clf.weights[j];
                }
                starts[i] = start;
            } else {
                starts[0] = 0;
            }
        }

        int[] ends = new int[this.clf.nRows];
        for (int i = 0; i < this.clf.nRows; i++) {
            ends[i] = this.clf.weights[i] + starts[i];
        }

        if (this.clf.nClasses == 2) {
            for (int i = 0; i < kernels.length; i++) {
                kernels[i] = -kernels[i];
            }
            double decision = 0.;
            for (int k = starts[1]; k < ends[1]; k++) {
                decision += kernels[k] * this.clf.coefficients[0][k];
            }
            for (int k = starts[0]; k < ends[0]; k++) {
                decision += kernels[k] * this.clf.coefficients[0][k];
            }
            decision += this.clf.intercepts[0];
            if (decision > 0) {
                return 0;
            }
            return 1;
        }

        double[] decisions = new double[this.clf.intercepts.length];
        for (int i = 0, d = 0, l = this.clf.nRows; i < l; i++) {
            for (int j = i + 1; j < l; j++) {
                double tmp = 0.;
                for (int k = starts[j]; k < ends[j]; k++) {
                    tmp += this.clf.coefficients[i][k] * kernels[k];
                }
                for (int k = starts[i]; k < ends[i]; k++) {
                    tmp += this.clf.coefficients[j - 1][k] * kernels[k];
                }
                decisions[d] = tmp + this.clf.intercepts[d];
                d++;
            }
        }

        int[] votes = new int[this.clf.intercepts.length];
        for (int i = 0, d = 0, l = this.clf.nRows; i < l; i++) {
            for (int j = i + 1; j < l; j++) {
                votes[d] = decisions[d] > 0 ? i : j;
                d++;
            }
        }

        int[] amounts = new int[this.clf.nClasses];
        for (int i = 0, l = votes.length; i < l; i++) {
            amounts[votes[i]] += 1;
        }

        int classVal = -1, classIdx = -1;
        for (int i = 0, l = amounts.length; i < l; i++) {
            if (amounts[i] > classVal) {
                classVal = amounts[i];
                classIdx = i;
            }
        }
        return this.clf.classes[classIdx];
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
            NuSVC clf = new NuSVC(modelData);

            // Prediction:
            int prediction = clf.predict(features);
            System.out.println(prediction);

        }
    }
}
"""
