# %% [markdown]
# # Command-line interface
#
# This example ports a trained classifier of scikit-learn to JavaScript with
# the command `porter` of
# [sklearn-porter](https://github.com/nok/sklearn-porter), which is available
# after the installation. The lines which start with `!` are executed in a
# shell.
#
# Running the ported classifier requires `node` for JavaScript.
#
# ## Show the usage
#
# The command has the subcommands `show`, `port` and `save`:

# %%
# !porter -h

# %% [markdown]
# Show the help of a subcommand with `-h`:

# %%
# !porter port -h

# %% [markdown]
# The subcommand `show` lists the supported estimators and transformers with
# the templates of a language:

# %%
# !porter show -l js

# %% [markdown]
# ## Train and save a classifier
#
# To port an estimator (or transformer or pipeline), save it to a file first
# (see [model persistence](https://scikit-learn.org/stable/model_persistence.html)).
# The files of this example are saved in `/tmp`:

# %%
# %cd /tmp

# %%
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

dump(clf, 'estimator.joblib')

# %% [markdown]
# ## Port the classifier
#
# The subcommand `port` prints the source code:

# %%
# !porter port estimator.joblib -l js -t attached

# %% [markdown]
# Redirect the output to save the source code to a file:

# %%
# !porter port estimator.joblib -l js -t attached > estimator.js

# %% [markdown]
# Run the ported classifier with the features of a sample:

# %%
# !node estimator.js 6.3 3.3 6.0 2.5

# %% [markdown]
# ## Save the classifier
#
# The subcommand `save` saves the source code to a file, with the template
# `exported` also the model data to a separate JSON file:

# %%
# !porter save estimator.joblib -l js -t exported --directory /tmp

# %% [markdown]
# ## Pass the output to other tools
#
# The output of `port` can be passed to other tools, e.g. to
# [terser](https://github.com/terser/terser) (with `npx` of npm), to minify
# the code:
#
# ```bash
# porter port estimator.joblib -l js -t attached | npx terser --compress -o estimator.min.js
# ```
