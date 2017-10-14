# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.naive_bayes import BernoulliNB
from sklearn_porter import Porter


iris_data = load_iris()
X, y = iris_data.data, iris_data.target

clf = BernoulliNB()
clf.fit(X, y)

porter = Porter(clf)
output = porter.export()
print(output)

"""
class Brain {

    private double[] priors;
    private double[][] negProbs;
    private double[][] delProbs;

    public Brain(double[] priors, double[][] negProbs, double[][] delProbs) {
        this.priors = priors;
        this.negProbs = negProbs;
        this.delProbs = delProbs;
    }

    public int predict(double[] features) {
        if (features.length != 4) return -1;
    
        double[] jll = new double[3];
        for (int i = 0; i < 3; i++) {
            double sum = 0.;
            for (int j = 0; j < 4; j++) {
                sum += features[i] * this.delProbs[j][i];
            }
            jll[i] = sum;
        }
        for (int i = 0; i < 3; i++) {
            double sum = 0.;
            for (int j = 0; j < 4; j++) {
                sum += this.negProbs[i][j];
            }
            jll[i] += this.priors[i] + sum;
        }
    
        double highestLikeli = Double.NEGATIVE_INFINITY;
        int classIndex = -1;
        for (int i = 0; i < 3; i++) {
            if (jll[i] > highestLikeli) {
                highestLikeli = jll[i];
                classIndex = i;
            }
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
            Brain brain = new Brain(priors, negProbs, delProbs);
            int estimation = brain.predict(features);
            System.out.println(estimation);

        }
    }
}
"""
