# %% [markdown]
# # Transformers
#
# Many models need preprocessed features, e.g. scaled features. Then the
# preprocessing has to run in the target language too. This example ports a
# `StandardScaler` to JavaScript with
# [sklearn-porter](https://github.com/nok/sklearn-porter) and runs it with
# Node.js.
#
# Running the ported transformer requires `node`.
#
# ## Train a transformer
#
# The features of the wine dataset have very different scales, e.g. the
# alcohol (about 13) and the proline (about 1000):

# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=3, suppress=True)

X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
scaler.fit(X)

print(X[:2])

# %% [markdown]
# ## Port the transformer
#
# The class `Porter` (and the functions `port`, `save`, `make` and `test`)
# accepts transformers like classifiers. Save the ported scaler:

# %%
from sklearn_porter import Porter

porter = Porter(scaler, language='js')
src_path = porter.save(directory='/tmp')

print(src_path)

# %% [markdown]
# ## Run the ported transformer
#
# The ported transformer prints the transformed features of each sample in
# one line, separated by spaces:

# %%
# !node /tmp/StandardScaler.js 14.23 1.71 2.43 15.6 127 2.8 3.06 0.28 2.29 5.64 1.04 3.92 1065

# %% [markdown]
# `transform` runs the ported transformer from Python, like `transform` of
# scikit-learn:

# %%
print(porter.transform(X[:2]))
print(scaler.transform(X[:2]))

# %% [markdown]
# ## Check the ported transformer
#
# The score is the fraction of samples whose transformed features match
# those of scikit-learn:

# %%
print(porter.test(X))

# %% [markdown]
# ## Combine transformers
#
# A `ColumnTransformer` (or a `FeatureUnion`) combines several transformers,
# also transformers which change the number of features. Here the first 12
# features are scaled and the proline is split into 3 bins (one-hot encoded),
# so a sample has 15 transformed features:

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer

columns = ColumnTransformer(
    [
        ('scale', StandardScaler(), slice(0, 12)),
        ('bins', KBinsDiscretizer(n_bins=3, encode='onehot-dense'), [12]),
    ]
)
columns.fit(X)

porter = Porter(columns, language='js')

print(porter.transform(X[:2]))
print(porter.test(X))

# %% [markdown]
# The [README](https://github.com/nok/sklearn-porter#transformers) lists all
# supported transformers.
#
# ## Next steps
#
# - [Pipelines](pipelines.ipynb): port the preprocessing and the classifier
#   of a pipeline to one source file.
