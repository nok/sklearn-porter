# %% [markdown]
# # Getting Started
#
# This example trains a small decision tree, ports it to JavaScript with
# [sklearn-porter](https://github.com/nok/sklearn-porter), runs it with
# Node.js and checks that it predicts the same classes as scikit-learn.
#
# Running the ported classifier requires `node`.
#
# ## Train a classifier

# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)

# %% [markdown]
# ## Port the classifier
#
# `port` returns the source code. With the template `combined` the splits of
# the tree become nested if-else statements:

# %%
from sklearn_porter import port

print(port(clf, language='js', template='combined'))

# %% [markdown]
# ## Run the ported classifier
#
# Save the source code to a file and pass the features of a sample:

# %%
from sklearn_porter import save

save(clf, language='js', template='combined', directory='/tmp')

# %%
# !node /tmp/DecisionTreeClassifier.js 6.3 3.3 6.0 2.5

# %% [markdown]
# Or use the class in your own code, e.g. with `require` in Node.js or with
# a `<script>` tag in a browser:

# %%
# !node -e "const Tree = require('/tmp/DecisionTreeClassifier.js'); console.log('Class: ' + new Tree().predict([6.3, 3.3, 6.0, 2.5]))"

# %% [markdown]
# ## Make predictions from Python
#
# `make` runs the ported classifier with several samples and returns the
# predicted classes and the probabilities:

# %%
from sklearn_porter import make

y_pred, y_proba = make(clf, X[[0, 50, 100]], language='js')

print(y_pred)
print(y_proba)

# %% [markdown]
# ## Check the ported classifier
#
# `test` compares the predictions of the ported classifier with those of
# scikit-learn. The score is the fraction of equal predictions, so `1.0`
# means that all predictions match. Always check the ported classifier with
# representative samples:

# %%
from sklearn_porter import test

print(test(clf, X, language='js'))

# %% [markdown]
# ## Templates and the class `Porter`
#
# The template defines how the model data is stored: `attached` (as arrays in
# the code), `combined` (in the logic of the code) or `exported` (in a
# separate JSON file, so you can update the model without changing the
# code). The class `Porter` keeps the language and the template for several
# calls:

# %%
from pathlib import Path

from sklearn_porter import Porter

porter = Porter(clf, language='js', template='exported')
src_path, json_path = porter.save(directory='/tmp')

print(Path(json_path).read_text())
print(porter.test(X))

# %% [markdown]
# ## Next steps
#
# - [Transformers](transformers.ipynb): port the preprocessing, e.g. a
#   `StandardScaler`.
# - [Pipelines](pipelines.ipynb): port the preprocessing and the classifier
#   of a pipeline to one source file.
# - [CLI](cli.ipynb): port a saved classifier with the command `porter`.
