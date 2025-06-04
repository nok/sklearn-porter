# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## NuSVC
#
# Documentation: [sklearn.svm.NuSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)

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

clf = svm.NuSVC(gamma=0.001, kernel='rbf', random_state=0)
clf.fit(X, y)

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
# with open('nusvc.c', 'w') as f:
#     f.write(output)

# Compile model:
# $ gcc nusvc.c -std=c99 -lm -o nusvc

# Run classification:
# $ ./nusvc 1 2 3 4
