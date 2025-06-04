# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## LinearSVC
#
# Documentation: [sklearn.svm.LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

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
from sklearn import svm

clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

# %% [markdown]
# ### Transpile classifier

# %%
from sklearn_porter import Porter

porter = Porter(clf, language='js')
output = porter.export()

print(output)

# %% [markdown]
# ### Run classification in JavaScript

# %%
# Save classifier:
# with open('LinearSVC.js', 'w') as f:
#     f.write(output)

# Run classification:
# if hash node 2/dev/null; then
#     node LinearSVC.js 1 2 3 4
# fi
