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
output = porter.export()
print(output)

"""
class BernoulliNB {

    private double[] priors;
    private double[][] negProbs;
    private double[][] delProbs;

    public BernoulliNB(double[] priors, double[][] negProbs, double[][] delProbs) {
        this.priors = priors;
        this.negProbs = negProbs;
        this.delProbs = delProbs;
    }

    public int predict(double[] features) {
        int nClasses = this.priors.length;
        int nFeatures = this.delProbs.length;
    
        double[] jll = new double[nClasses];
        for (int i = 0; i < nClasses; i++) {
            double sum = 0.;
            for (int j = 0; j < nFeatures; j++) {
                sum += features[i] * this.delProbs[j][i];
            }
            jll[i] = sum;
        }
        for (int i = 0; i < nClasses; i++) {
            double sum = 0.;
            for (int j = 0; j < nFeatures; j++) {
                sum += this.negProbs[i][j];
            }
            jll[i] += this.priors[i] + sum;
        }
    
        int classIndex = 0;
        for (int i = 0; i < nClasses; i++) {
            classIndex = jll[i] > jll[classIndex] ? i : classIndex;
        }
        return classIndex;
    }

    public static void main(String[] args) {
        if (args.length == 4) {

            // Features:
            double[] features = new double[args.length];
            for (int i = 0, l = args.length; i < l; i++) {
                features[i] = Double.parseDouble(args[i]);
            }

            // Parameters:
            final double[] priors = {-1.0986122886681096, -1.0986122886681096, -1.0986122886681096};
            final double[][] negProbs = {{-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138}, {-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138}, {-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138}};
            final double[][] delProbs = {{3.931825632724312, 3.931825632724312, 3.931825632724312}, {3.931825632724312, 3.931825632724312, 3.931825632724312}, {3.931825632724312, 3.931825632724312, 3.931825632724312}, {3.931825632724312, 3.931825632724312, 3.931825632724312}};

            // Prediction:
            BernoulliNB clf = new BernoulliNB(priors, negProbs, delProbs);
            int estimation = clf.predict(features);
            System.out.println(estimation);

        }
    }
}
"""
