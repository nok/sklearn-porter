from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from onl.nok.sklearn.Porter import port

iris = load_iris()
base_estimator = DecisionTreeClassifier(max_depth=4, random_state=0)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=0)
clf.fit(iris.data, iris.target)

# Cheese!

print(port(clf))

# class Tmp {
#     public static float[] predict_000(float[] atts) {
#         int n_classes = 3;
#         float[] classes = new float[n_classes];
#
#         if (atts[3] <= 0.800000011920929f) {
#             classes[0] = 0.333333333333333f;
#             classes[1] = 0.0f;
#             classes[2] = 0.0f;
#         } else {
#             if (atts[3] <= 1.75f) {
#                 if (atts[2] <= 4.949999809265137f) {
#                     if (atts[3] <= 1.650000095367432f) {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.313333333333333f;
#                         classes[2] = 0.0f;
#                     } else {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.0f;
#                         classes[2] = 0.006666666666666667f;
#                     }
#                 } else {
#                     if (atts[3] <= 1.549999952316284f) {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.0f;
#                         classes[2] = 0.02f;
#                     } else {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.01333333333333333f;
#                         classes[2] = 0.006666666666666667f;
#                     }
#                 }
#             } else {
#                 if (atts[2] <= 4.850000381469727f) {
#                     if (atts[0] <= 5.949999809265137f) {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.006666666666666667f;
#                         classes[2] = 0.0f;
#                     } else {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.0f;
#                         classes[2] = 0.01333333333333333f;
#                     }
#                 } else {
#                     classes[0] = 0.0f;
#                     classes[1] = 0.0f;
#                     classes[2] = 0.2866666666666664f;
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
#         if (atts[2] <= 5.149999618530273f) {
#             if (atts[2] <= 2.450000047683716f) {
#                 classes[0] = 8.32907244640284e-05f;
#                 classes[1] = 0.0f;
#                 classes[2] = 0.0f;
#             } else {
#                 if (atts[3] <= 1.75f) {
#                     if (atts[0] <= 4.949999809265137f) {
#                         classes[0] = 0.0f;
#                         classes[1] = 1.665814489280568e-06f;
#                         classes[2] = 1.665814489280568e-06f;
#                     } else {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.499954190101545f;
#                         classes[2] = 3.331628978561136e-06f;
#                     }
#                 } else {
#                     if (atts[1] <= 3.150000095367432f) {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.0f;
#                         classes[2] = 1.998977387136681e-05f;
#                     } else {
#                         classes[0] = 0.0f;
#                         classes[1] = 1.665814489280568e-06f;
#                         classes[2] = 1.665814489280568e-06f;
#                     }
#                 }
#             }
#         } else {
#             classes[0] = 0.0f;
#             classes[1] = 0.0f;
#             classes[2] = 0.4999325345131842f;
#         }
#
#         return classes;
#     }
#
#     public static float[] predict_002(float[] atts) {
#         int n_classes = 3;
#         float[] classes = new float[n_classes];
#
#         if (atts[3] <= 1.549999952316284f) {
#             if (atts[2] <= 4.949999809265137f) {
#                 if (atts[3] <= 0.800000011920929f) {
#                     classes[0] = 2.678817718645179e-08f;
#                     classes[1] = 0.0f;
#                     classes[2] = 0.0f;
#                 } else {
#                     classes[0] = 0.0f;
#                     classes[1] = 0.0001847310949932946f;
#                     classes[2] = 0.0f;
#                 }
#             } else {
#                 classes[0] = 0.0f;
#                 classes[1] = 0.0f;
#                 classes[2] = 0.4996966431023263f;
#             }
#         } else {
#             if (atts[2] <= 5.149999618530273f) {
#                 if (atts[3] <= 1.849999904632568f) {
#                     if (atts[1] <= 2.599999904632568f) {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.0f;
#                         classes[2] = 0.0001114730152488703f;
#                     } else {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.4997348575020662f;
#                         classes[2] = 2.678817718645176e-09f;
#                     }
#                 } else {
#                     classes[0] = 0.0f;
#                     classes[1] = 0.0f;
#                     classes[2] = 0.0001114767655936764f;
#                 }
#             } else {
#                 classes[0] = 0.0f;
#                 classes[1] = 0.0f;
#                 classes[2] = 0.0001607890527769535f;
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
#         if (atts[3] <= 1.75f) {
#             if (atts[3] <= 1.549999952316284f) {
#                 if (atts[2] <= 4.949999809265137f) {
#                     if (atts[2] <= 2.450000047683716f) {
#                         classes[0] = 9.257653973762734e-11f;
#                         classes[1] = 0.0f;
#                         classes[2] = 0.0f;
#                     } else {
#                         classes[0] = 0.0f;
#                         classes[1] = 6.384072136521276e-07f;
#                         classes[2] = 0.0f;
#                     }
#                 } else {
#                     classes[0] = 0.0f;
#                     classes[1] = 0.0f;
#                     classes[2] = 0.001726888164690719f;
#                 }
#             } else {
#                 if (atts[0] <= 6.949999809265137f) {
#                     if (atts[1] <= 2.599999904632568f) {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.0f;
#                         classes[2] = 3.852365897848193e-07f;
#                     } else {
#                         classes[0] = 0.0f;
#                         classes[1] = 0.4990242342550203f;
#                         classes[2] = 0.0f;
#                     }
#                 } else {
#                     classes[0] = 0.0f;
#                     classes[1] = 0.0f;
#                     classes[2] = 5.556073060838475e-07f;
#                 }
#             }
#         } else {
#             if (atts[1] <= 3.150000095367432f) {
#                 classes[0] = 0.0f;
#                 classes[1] = 0.0f;
#                 classes[2] = 0.4991355736414027f;
#             } else {
#                 if (atts[2] <= 4.949999809265137f) {
#                     classes[0] = 0.0f;
#                     classes[1] = 0.0001113393363919567f;
#                     classes[2] = 0.0f;
#                 } else {
#                     classes[0] = 0.0f;
#                     classes[1] = 0.0f;
#                     classes[2] = 3.852588081543566e-07f;
#                 }
#             }
#         }
#
#         return classes;
#     }
#
#     public static int predict(float[] atts) {
#         int n_estimators = 4;
#         int n_classes = 3;
#
#         float[][] preds = new float[n_estimators][];
#         preds[0] = Tmp.predict_000(atts);
#         preds[1] = Tmp.predict_001(atts);
#         preds[2] = Tmp.predict_002(atts);
#         preds[3] = Tmp.predict_003(atts);
#
#         int i, j;
#         float normalizer, sum;
#         for (i = 0; i < n_estimators; i++) {
#             normalizer = 0.f;
#             for (j = 0; j < n_classes; j++) {
#                 normalizer += preds[i][j];
#             }
#             if (normalizer == 0.f) {
#                 normalizer = 1.0f;
#             }
#             for (j = 0; j < n_classes; j++) {
#                 preds[i][j] = preds[i][j] / normalizer;
#                 if (preds[i][j] < 0.000000000000000222044604925f) {
#                     preds[i][j] = 0.000000000000000222044604925f;
#                 }
#                 preds[i][j] = (float) Math.log(preds[i][j]);
#             }
#             sum = 0.0f;
#             for (j = 0; j < n_classes; j++) {
#                 sum += preds[i][j];
#             }
#             for (j = 0; j < n_classes; j++) {
#                 preds[i][j] = (n_classes - 1) * (preds[i][j] - (1.f / n_classes) * sum);
#             }
#         }
#         float[] classes = new float[n_classes];
#         for (i = 0; i < n_estimators; i++) {
#             for (j = 0; j < n_classes; j++) {
#                 classes[j] += preds[i][j];
#             }
#         }
#         int idx = 0;
#         float val = Float.NEGATIVE_INFINITY;
#         for (i = 0; i < n_classes; i++) {
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