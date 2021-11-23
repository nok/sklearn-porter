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
import numpy as np

# transfer single-output into multi-labels
y_multi_label = []
for x in iris_data.target:
    if x == 0:
        y_multi_label.append([1,1,0])
    elif x == 1:
        y_multi_label.append([0,1,1])
    else:
        y_multi_label.append([1,0,1])
y = np.array(y_multi_label)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_data.data, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# %% [markdown]
# ### Transpile classifier

# %%
from sklearn_porter import Porter

porter = Porter(clf, language='c')
output = porter.export()

print(output)

# %% [markdown]
# ### Run classification in C

# %%
# Save model:
# with open('tree.c', 'w') as f:
#     f.write(output)

# Compile model:
# $ gcc tree.c -std=c99 -lm -o tree

# Run classification:
# $ ./tree 1 2 3 4
