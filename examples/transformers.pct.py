# %% [markdown]
# # Transformers
#
# This example ports trained transformers of scikit-learn (e.g. for the
# preprocessing) with [sklearn-porter](https://github.com/nok/sklearn-porter).
# A ported transformer has the method `transform`, which returns the
# transformed features of a sample, analogous to `predict` of a ported
# estimator. The functions `port`, `save`, `make` and `test` accept
# transformers too.
#
# Transforming features with the ported transformer requires a compiler or
# interpreter of the target language, here `node` for JavaScript.
#
# ## Train a transformer
#
# Load the data and train a transformer:

# %%
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
scaler.fit(X)

# %% [markdown]
# ## Port the transformer
#
# **Step 1**: Port (transpile) the transformer and get the source code:

# %%
from sklearn_porter import Transformer

transformer = Transformer(scaler, language='js', template='attached')
output = transformer.port()

print(output)

# %% [markdown]
# **Step 2**: Save the ported transformer:

# %%
src_path = transformer.save(directory='/tmp')

print(src_path)

# %% [markdown]
# **Step 3**: Transform the features with the ported transformer:

# %%
X_tr = transformer.make(X[::30])

print(X_tr)

# %% [markdown]
# **Step 4**: Always check the ported transformer: the score is the fraction
# of samples whose transformed features match those of the original
# transformer:

# %%
score = transformer.test(X[::15])

print(score)

# %% [markdown]
# The class `Porter` and the functions `make` and `test` accept transformers
# too:

# %%
from sklearn_porter import Porter, make, test

X_tr = Porter(scaler, language='js').transform(X[::30])
X_tr = make(scaler, X[::30], language='js')
score = test(scaler, X[::15], language='js')

print(score)

# %% [markdown]
# ## Use the ported transformer
#
# The ported transformer reads the features of one sample from its arguments
# or of several samples (one per line) from the standard input. It prints the
# transformed features of each sample in one line (separated by spaces),
# which is also the input format of the ported estimators:

# %%
# !node /tmp/StandardScaler.js 5.1 3.5 1.4 0.2

# %% [markdown]
# ## Change the number of features
#
# Transformers can change the number of features, e.g. a `KBinsDiscretizer`
# with a one-hot encoding of the bins. A `ColumnTransformer` (or a
# `FeatureUnion`) combines the transformed features of several transformers,
# if the transformers of all its parts are supported:

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer

columns = ColumnTransformer(
    [
        ('scale', StandardScaler(), [0, 1]),
        ('bins', KBinsDiscretizer(n_bins=3, encode='onehot-dense'), [2, 3]),
    ]
)
columns.fit(X)

transformer = Transformer(columns, language='js')
X_tr = transformer.make(X[::30])

print(X_tr)
print(transformer.test(X[::15]))

# %% [markdown]
# The [README](https://github.com/nok/sklearn-porter#transformers) lists all
# supported transformers, the
# [estimator notes](https://github.com/nok/sklearn-porter/blob/main/docs/estimator-notes.md)
# describe the parameters of single transformers which can't be ported.
#
# ## Next steps
#
# - [Pipelines](pipelines.ipynb): port the preprocessing and the prediction
#   of a pipeline to one source file.
