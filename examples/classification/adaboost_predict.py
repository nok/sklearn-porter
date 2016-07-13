from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from onl.nok.sklearn.export.Export import Export

iris = load_iris()
base_estimator = DecisionTreeClassifier(max_depth=4, random_state=0)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=0)
clf.fit(iris.data, iris.target)

clf.predict([[5.7, 2.8, 4.1, 1.3]])

# Cheese!

trees = Export.export(clf)
# print(trees)

# class Tmp {
#     public static float[] predict_000(float[] atts) {
#         int n_classes = 3;
#         float[] classes = new float[n_classes];
#
#         if (atts[3] <= 0.800000f) {
#             classes[0] = 0.333333f;
#             classes[1] = 0.000000f;
#             classes[2] = 0.000000f;
#         } else {
#             if (atts[3] <= 1.750000f) {
#                 if (atts[2] <= 4.950000f) {
#                     if (atts[3] <= 1.650000f) {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.313333f;
#                         classes[2] = 0.000000f;
#                     } else {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000000f;
#                         classes[2] = 0.006667f;
#                     }
#                 } else {
#                     if (atts[3] <= 1.550000f) {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000000f;
#                         classes[2] = 0.020000f;
#                     } else {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.013333f;
#                         classes[2] = 0.006667f;
#                     }
#                 }
#             } else {
#                 if (atts[2] <= 4.850000f) {
#                     if (atts[0] <= 5.950000f) {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.006667f;
#                         classes[2] = 0.000000f;
#                     } else {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000000f;
#                         classes[2] = 0.013333f;
#                     }
#                 } else {
#                     classes[0] = 0.000000f;
#                     classes[1] = 0.000000f;
#                     classes[2] = 0.286667f;
#                 }
#             }
#         }
#
#         return classes;
#     }
#
#     public static float[] predict_001(float[] atts) {
#         int n_classes = 3;
#         float[] classes = new float[n_classes];
#
#         if (atts[2] <= 5.150000f) {
#             if (atts[2] <= 2.450000f) {
#                 classes[0] = 0.000083f;
#                 classes[1] = 0.000000f;
#                 classes[2] = 0.000000f;
#             } else {
#                 if (atts[3] <= 1.750000f) {
#                     if (atts[0] <= 4.950000f) {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000002f;
#                         classes[2] = 0.000002f;
#                     } else {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.499954f;
#                         classes[2] = 0.000003f;
#                     }
#                 } else {
#                     if (atts[1] <= 3.150000f) {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000000f;
#                         classes[2] = 0.000020f;
#                     } else {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000002f;
#                         classes[2] = 0.000002f;
#                     }
#                 }
#             }
#         } else {
#             classes[0] = 0.000000f;
#             classes[1] = 0.000000f;
#             classes[2] = 0.499933f;
#         }
#
#         return classes;
#     }
#
#     public static float[] predict_002(float[] atts) {
#         int n_classes = 3;
#         float[] classes = new float[n_classes];
#
#         if (atts[3] <= 1.550000f) {
#             if (atts[2] <= 4.950000f) {
#                 if (atts[3] <= 0.800000f) {
#                     classes[0] = 0.000000f;
#                     classes[1] = 0.000000f;
#                     classes[2] = 0.000000f;
#                 } else {
#                     classes[0] = 0.000000f;
#                     classes[1] = 0.000185f;
#                     classes[2] = 0.000000f;
#                 }
#             } else {
#                 classes[0] = 0.000000f;
#                 classes[1] = 0.000000f;
#                 classes[2] = 0.499697f;
#             }
#         } else {
#             if (atts[2] <= 5.150000f) {
#                 if (atts[3] <= 1.850000f) {
#                     if (atts[1] <= 2.600000f) {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000000f;
#                         classes[2] = 0.000111f;
#                     } else {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.499735f;
#                         classes[2] = 0.000000f;
#                     }
#                 } else {
#                     classes[0] = 0.000000f;
#                     classes[1] = 0.000000f;
#                     classes[2] = 0.000111f;
#                 }
#             } else {
#                 classes[0] = 0.000000f;
#                 classes[1] = 0.000000f;
#                 classes[2] = 0.000161f;
#             }
#         }
#
#         return classes;
#     }
#
#     public static float[] predict_003(float[] atts) {
#         int n_classes = 3;
#         float[] classes = new float[n_classes];
#
#         if (atts[3] <= 1.750000f) {
#             if (atts[3] <= 1.550000f) {
#                 if (atts[2] <= 4.950000f) {
#                     if (atts[2] <= 2.450000f) {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000000f;
#                         classes[2] = 0.000000f;
#                     } else {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000001f;
#                         classes[2] = 0.000000f;
#                     }
#                 } else {
#                     classes[0] = 0.000000f;
#                     classes[1] = 0.000000f;
#                     classes[2] = 0.001727f;
#                 }
#             } else {
#                 if (atts[2] <= 5.450000f) {
#                     if (atts[1] <= 2.600000f) {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.000000f;
#                         classes[2] = 0.000000f;
#                     } else {
#                         classes[0] = 0.000000f;
#                         classes[1] = 0.499024f;
#                         classes[2] = 0.000000f;
#                     }
#                 } else {
#                     classes[0] = 0.000000f;
#                     classes[1] = 0.000000f;
#                     classes[2] = 0.000001f;
#                 }
#             }
#         } else {
#             if (atts[1] <= 3.150000f) {
#                 classes[0] = 0.000000f;
#                 classes[1] = 0.000000f;
#                 classes[2] = 0.499136f;
#             } else {
#                 if (atts[2] <= 4.950000f) {
#                     classes[0] = 0.000000f;
#                     classes[1] = 0.000111f;
#                     classes[2] = 0.000000f;
#                 } else {
#                     classes[0] = 0.000000f;
#                     classes[1] = 0.000000f;
#                     classes[2] = 0.000000f;
#                 }
#             }
#         }
#
#         return classes;
#     }
#
#     public static int predict(float[] atts) {
#         int n_estimators = 4;
#         float[][] preds = new float[n_estimators][];
#         preds[0] = Tmp.predict_000(atts);
#         preds[1] = Tmp.predict_001(atts);
#         preds[2] = Tmp.predict_002(atts);
#         preds[3] = Tmp.predict_003(atts);
#         int n_classes = 3;
#         float[] classes = new float[n_classes];
#         for (int i = 1; i < n_estimators; i++) {
#             for (int j = 0; j < n_classes; j++) {
#                 classes[j] += preds[i][j];
#             }
#         }
#
#         int idx = 0;
#         float val = classes[0];
#         for (int i = 1; i < n_classes; i++) {
#             if (classes[i] > val) {
#                 idx = i;
#             }
#         }
#         return idx;
#     }
#
#     public static void main(String[] args) {
#
#         int pred = Tmp.predict(new float[]{0.3f, 0.5f, 0f, 0.5f});
#         System.out.println(pred);
#
#         if (args.length == 4) {
#             float[] atts = new float[args.length];
#             for (int i = 0; i < args.length; i++) {
#                 atts[i] = Float.parseFloat(args[i]);
#             }
#             System.out.println(Tmp.predict(atts));
#         }
#     }
# }