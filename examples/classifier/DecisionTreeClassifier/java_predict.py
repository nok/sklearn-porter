from sklearn.tree import tree
from sklearn.datasets import load_iris

from onl.nok.sklearn.Porter import port

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

# Cheese!

print(port(clf))

# class Tmp {
#     public static int predict(float[] atts) {
#         int n_classes = 3;
#         int[] classes = new int[n_classes];
#
#         if (atts[2] <= 2.450000f) {
#             classes[0] = 50;
#             classes[1] = 0;
#             classes[2] = 0;
#         } else {
#             if (atts[3] <= 1.750000f) {
#                 if (atts[2] <= 4.950000f) {
#                     if (atts[3] <= 1.650000f) {
#                         classes[0] = 0;
#                         classes[1] = 47;
#                         classes[2] = 0;
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 1;
#                     }
#                 } else {
#                     if (atts[3] <= 1.550000f) {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 3;
#                     } else {
#                         if (atts[2] <= 5.450000f) {
#                             classes[0] = 0;
#                             classes[1] = 2;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 1;
#                         }
#                     }
#                 }
#             } else {
#                 if (atts[2] <= 4.850000f) {
#                     if (atts[1] <= 3.100000f) {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 2;
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 1;
#                         classes[2] = 0;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 43;
#                 }
#             }
#         }
#
#         int idx = 0;
#         int val = classes[0];
#         for (int i = 1; i < n_classes; i++) {
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
#             for (int i = 0; i < args.length; i++) {
#                 atts[i] = Float.parseFloat(args[i]);
#             }
#             System.out.println(Tmp.predict(atts));
#         }
#     }
# }