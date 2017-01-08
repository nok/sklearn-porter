from sklearn.datasets import load_iris
from sklearn.naive_bayes import BernoulliNB

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
clf = BernoulliNB()
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

        double[] priors = {-1.0986122886681096, -1.0986122886681096, -1.0986122886681096};
        double[][] negProbs = {{-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138}, {-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138}, {-3.9512437185814138, -3.9512437185814138, -3.9512437185814138, -3.9512437185814138}};
        double[][] delProbs = {{3.931825632724312, 3.931825632724312, 3.931825632724312}, {3.931825632724312, 3.931825632724312, 3.931825632724312}, {3.931825632724312, 3.931825632724312, 3.931825632724312}, {3.931825632724312, 3.931825632724312, 3.931825632724312}};

        double[] jll = new double[3];
        for (i = 0; i < 3; i++) {
            double sum = 0.;
            for (j = 0; j < 4; j++) {
                sum += atts[i] * delProbs[j][i];
            }
            jll[i] = sum;
        }
        for (i = 0; i < 3; i++) {
            double sum = 0.;
            for (j = 0; j < 4; j++) {
                sum += negProbs[i][j];
            }
            jll[i] += priors[i] + sum;
        }

        double highestLikeli = Double.NEGATIVE_INFINITY;
        int classIndex = -1;
        for (i = 0; i < 3; i++) {
            if (jll[i] > highestLikeli) {
                highestLikeli = jll[i];
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