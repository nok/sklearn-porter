# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## RandomForestClassifier
#
# Documentation: [sklearn.ensemble.RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

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
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=15, max_depth=None,
                             min_samples_split=2, random_state=0)
clf.fit(X, y)

# %% [markdown]
# ### Transpile classifier

# %%
from sklearn_porter import Porter

porter = Porter(clf, language='c')
output = porter.export(embed_data=True)

print(output)

# %% [markdown]
# ### Run classification in C

# %%
# Save model:
# with open('forest.c', 'w') as f:
#     f.write(output)

# Compile model:
# $ gcc forest.c -std=c99 -lm -o forest

# Run classification:
# $ ./forest 1 2 3 4
