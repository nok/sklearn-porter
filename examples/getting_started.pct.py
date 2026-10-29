# %% [markdown]
# # Getting Started
#
# This example ports a trained classifier of scikit-learn to JavaScript and
# Java with [sklearn-porter](https://github.com/nok/sklearn-porter), makes
# predictions with the ported classifier and checks them.
#
# Making predictions with the ported classifier requires a compiler or
# interpreter of the target language, here `node` for JavaScript and the JDK
# for Java.
#
# ## Train a classifier
#
# Load the data and train a classifier:

# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X, y)

# %% [markdown]
# ## Functional API
#
# **Step 1**: Check whether the classifier can be ported to a language with a
# template. The template defines how the model data is stored in the
# generated code: `attached` (embedded in the code), `exported` (in a
# separate JSON file) or `combined` (compiled into the logic of the code):

# %%
from sklearn_porter import can, make, port, save, test

print(can(clf, language='js', template='attached'))

# %% [markdown]
# **Step 2**: Port (transpile) the classifier and get the source code:

# %%
output = port(clf, language='js', template='attached')

print(output)

# %% [markdown]
# With the template `combined` the model data is compiled into nested
# if-else statements:

# %%
output = port(clf, language='js', template='combined')

print(output)

# %% [markdown]
# **Step 3**: Save the ported classifier to files. With the template
# `exported` the model data is saved in a separate JSON file:

# %%
src_path, json_path = save(
    clf, language='js', template='exported', directory='/tmp'
)

print(src_path, json_path)

# %%
from pathlib import Path

print(Path(json_path).read_text())

# %% [markdown]
# **Step 4**: Make predictions with the ported classifier. `make` returns the
# predicted classes and the probabilities of the classes:

# %%
y_classes, y_probas = make(clf, X[::15], language='js', template='exported')

print(y_classes)
print(y_probas)

# %% [markdown]
# **Step 5**: Always check the ported classifier: the score is the fraction
# of predictions that match those of the original classifier:

# %%
score = test(clf, X[::15], language='js', template='exported')

print(score)

# %% [markdown]
# ## Object-oriented API
#
# The central class `Porter` accepts an estimator, a transformer or a
# pipeline and keeps the language and the template.
#
# **Step 1**: Port (transpile) the classifier:

# %%
from sklearn_porter import Porter

porter = Porter(clf, language='java', template='attached')
output = porter.port()

print(output)

# %% [markdown]
# **Step 2**: Change the template and save the ported classifier:

# %%
porter.template = 'exported'
src_path, json_path = porter.save(directory='/tmp')

print(src_path, json_path)

# %% [markdown]
# **Step 3**: Make predictions with the ported classifier:

# %%
y_classes, y_probas = porter.predict(X[::15])

print(y_classes)
print(y_probas)

# %% [markdown]
# **Step 4**: Always check the ported classifier:

# %%
score = porter.test(X[::15])

print(score)

# %% [markdown]
# ## Next steps
#
# - [Transformers](transformers.ipynb): port transformers, e.g. for the
#   preprocessing.
# - [Pipelines](pipelines.ipynb): port the preprocessing and the prediction
#   of a pipeline to one source file.
