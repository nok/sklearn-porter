# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## ExtraTreesClassifier
#
# Documentation: [sklearn.ensemble.ExtraTreesClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

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
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=15, random_state=0)
clf.fit(X, y)

# %% [markdown]
# ### Transpile classifier

# %%
from sklearn_porter import Porter

porter = Porter(clf, language='java')
output = porter.export(embed_data=True)

print(output)

# %% [markdown]
# ### Run classification in Java

# %%
# Save classifier:
# with open('ExtraTreesClassifier.java', 'w') as f:
#     f.write(output)

# Compile model:
# $ javac -cp . ExtraTreesClassifier.java

# Run classification:
# $ java ExtraTreesClassifier 1 2 3 4
