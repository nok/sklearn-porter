# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## DecisionTreeClassifier
#
# Documentation: [sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

# %%
import sys
sys.path.append('../../../../..')

# %% [markdown]
# ### Load data

# %%
from sklearn.datasets import load_iris

iris_data = load_iris()

X = iris_data.data
y = iris_data.target

print(X.shape, y.shape)

# %% [markdown]
# ### Train classifier

# %%
from sklearn.tree import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# %% [markdown]
# ### Transpile classifier

# %%
from sklearn_porter import Porter

porter = Porter(clf, language='java')
output = porter.export(embed_data=True)

print(output)

# class DecisionTreeClassifier {
#
#     private static int findMax(int[] nums) {
#         int index = 0;
#         for (int i = 0; i < nums.length; i++) {
#             index = nums[i] > nums[index] ? i : index;
#         }
#         return index;
#     }
#
#     public static int predict(double[] features) {
#         int[] classes = new int[3];
#
#         if (features[3] <= 0.800000011920929) {
#             classes[0] = 50;
#             classes[1] = 0;
#             classes[2] = 0;
#         } else {
#             if (features[3] <= 1.75) {
#                 if (features[2] <= 4.950000047683716) {
#                     if (features[3] <= 1.6500000357627869) {
#                         classes[0] = 0;
#                         classes[1] = 47;
#                         classes[2] = 0;
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 1;
#                     }
#                 } else {
#                     if (features[3] <= 1.550000011920929) {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 3;
#                     } else {
#                         if (features[2] <= 5.450000047683716) {
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
#                 if (features[2] <= 4.8500001430511475) {
#                     if (features[0] <= 5.950000047683716) {
#                         classes[0] = 0;
#                         classes[1] = 1;
#                         classes[2] = 0;
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 2;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 43;
#                 }
#             }
#         }
#
#         return findMax(classes);
#     }
#
#     public static void main(String[] args) {
#         if (args.length == 4) {
#
#             // Features:
#             double[] features = new double[args.length];
#             for (int i = 0, l = args.length; i < l; i++) {
#                 features[i] = Double.parseDouble(args[i]);
#             }
#
#             // Prediction:
#             int prediction = DecisionTreeClassifier.predict(features);
#             System.out.println(prediction);
#
#         }
#     }
# }

# %% [markdown]
# ### Run classification in Java

# %%
# Save classifier:
# with open('DecisionTreeClassifier.java', 'w') as f:
#     f.write(output)

# Compile model:
# $ javac -cp . DecisionTreeClassifier.java

# Run classification:
# $ java DecisionTreeClassifier 1 2 3 4
