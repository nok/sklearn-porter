# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

from sklearn_porter import Porter


X, y = load_iris(return_X_y=True)
clf = GaussianNB()
clf.fit(X, y)

# Cheese!

result = Porter().port(clf)
# result = Porter(language='java').port(clf)
print(result)

"""
class Tmp {

    public static int predict(double[] atts) {
        if (atts.length != 4) {
            return -1;
        }
        int i, j;

        double[] priors = {0.33333333333333331, 0.33333333333333331, 0.33333333333333331};
        double[][] sigmas = {{0.12176400309242481, 0.14227600309242491, 0.029504003092424898, 0.011264003092424885}, {0.26110400309242499, 0.096500003092424902, 0.21640000309242502, 0.038324003092424869}, {0.39625600309242481, 0.10192400309242496, 0.29849600309242508, 0.073924003092424875}};
        double[][] thetas = {{5.0059999999999993, 3.4180000000000006, 1.464, 0.24399999999999991}, {5.9359999999999999, 2.7700000000000005, 4.2599999999999998, 1.3259999999999998}, {6.5879999999999983, 2.9739999999999998, 5.5519999999999996, 2.0259999999999998}};
        double[] likelihoods = new double[3];

        for (i = 0; i < 3; i++) {
            double sum = 0.;
            for (j = 0; j < 4; j++) {
                sum += Math.log(2. * Math.PI * sigmas[i][j]);
            }
            double nij = -0.5 * sum;
            sum = 0.;
            for (j = 0; j < 4; j++) {
                sum += Math.pow(atts[j] - thetas[i][j], 2.) / sigmas[i][j];
            }
            nij -= 0.5 * sum;
            likelihoods[i] = Math.log(priors[i]) + nij;
        }

        double highestLikeli = Double.NEGATIVE_INFINITY;
        int classIndex = -1;
        for (i = 0; i < 3; i++) {
            if (likelihoods[i] > highestLikeli) {
                highestLikeli = likelihoods[i];
                classIndex = i;
            }
        }
        return classIndex;
    }

    public static void main(String[] args) {
        if (args.length == 4) {
            double[] atts = new double[args.length];
            for (int i = 0, l = args.length; i < l; i++) {
                atts[i] = Double.parseDouble(args[i]);
            }
            System.out.println(Tmp.predict(atts));
        }
    }
}
"""