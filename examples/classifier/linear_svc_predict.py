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
#         if (atts.length != 4) {
#             return -1;
#         }
#         float[][] coefs = {{0.18424209458473811f, 0.45123000025163923f, -0.80794587716737576f, -0.45071660033253858f}, {0.052877455748516447f, -0.89214995228605254f, 0.40398084459610972f, -0.9376821661447452f}, {-0.85070784319293802f, -0.98670214922204336f, 1.381010448739191f, 1.8654095662423917f}};
#         float[] inters = {0.10956266406702335f, 1.6636707776739579f, -1.7096109416521363f};
#         float[] classes = new float[3];
#         for (int i = 0; i < 3; i++) {
#             float prob = 0.0f;
#             for (int j = 0; j < 4; j++) {
#                 prob += coefs[i][j] * atts[j];
#             }
#             classes[i] = prob + inters[i];
#         }
#         int idx = 0;
#         float val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > val) {
#                 idx = i;
#                 val = classes[i];
#             }
#         }
#         return idx;
#     }
#
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

