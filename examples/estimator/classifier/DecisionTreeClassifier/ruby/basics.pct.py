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

porter = Porter(clf, language='ruby')
output = porter.export()

print(output)
