from sklearn import svm
from sklearn.datasets import load_iris

from onl.nok.sklearn.Porter import port

iris = load_iris()
clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(iris.data, iris.target)

# Cheese!

print(port(clf))

# class Tmp {
#     public static int predict(float[] atts) {
#         if (atts.length != 4) { return -1; }
#         double[][] coefs = {{0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858}, {0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452}, {-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917}};
#         double[] inters = {0.10956266406702335, 1.6636707776739579, -1.7096109416521363};
#         int class_idx = -1;
#         double class_val = Double.NEGATIVE_INFINITY;
#         for (int i = 0; i < 3; i++) {
#             double prob = 0.;
#             for (int j = 0; j < 4; j++) {
#                 prob += coefs[i][j] * atts[j];
#             }
#             if (prob + inters[i] > class_val) {
#                 class_val = prob + inters[i];
#                 class_idx = i;
#             }
#         }
#         return class_idx;
#     }
#     public static void main(String[] args) {
#         if (args.length == 4) {
#             float[] atts = new float[args.length];
#             for (int i = 0, l = args.length; i < l; i++) {
#                 atts[i] = Float.parseFloat(args[i]);
#             }
#             System.out.println(Tmp.predict(atts));
#         }
#     }
# }
