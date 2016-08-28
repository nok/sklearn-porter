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
#         float[][] coefs = {
#                 {0.184242094585f, 0.451230000252f, -0.807945877167f, -0.450716600333f},
#                 {0.0528774557485f, -0.892149952286f, 0.403980844596f, -0.937682166145f},
#                 {-0.850707843193f, -0.986702149222f, 1.38101044874f, 1.86540956624f}
#         };
#         float[] inters = {0.109562664067f, 1.66367077767f, -1.70961094165f};
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
