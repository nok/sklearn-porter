# %% [markdown]
# # Pipelines
#
# This example ports a trained `Pipeline` of scikit-learn with
# [sklearn-porter](https://github.com/nok/sklearn-porter). A pipeline of
# supported transformers (the preprocessing) and a supported final estimator
# (the prediction) is ported to one source file: the ported final estimator,
# which transforms the features of each sample with the ported transformers
# before its prediction. The functions `port`, `save`, `make` and `test`
# accept pipelines too.
#
# Making predictions with the ported pipeline requires a compiler or
# interpreter of the target language, here `node` for JavaScript and the JDK
# for Java.
#
# ## Train a pipeline
#
# Load the data and train a pipeline:

# %%
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC())
pipe.fit(X, y)

# %% [markdown]
# ## Port the pipeline
#
# **Step 1**: Port (transpile) the pipeline and get the source code:

# %%
from sklearn_porter import make, port, save, test

output = port(pipe, language='js', template='attached')

print(output)

# %% [markdown]
# **Step 2**: Save the ported pipeline:

# %%
src_path = save(pipe, language='js', directory='/tmp')

print(src_path)

# %% [markdown]
# **Step 3**: Make predictions with the ported pipeline, like
# `pipe.predict(X[::15])`:

# %%
y_classes, _ = make(pipe, X[::15], language='js')

print(y_classes)

# %% [markdown]
# **Step 4**: Always check the ported pipeline: the score is the fraction of
# predictions that match those of the original pipeline:

# %%
score = test(pipe, X[::15], language='js')

print(score)

# %% [markdown]
# ## Use the ported pipeline
#
# The generated file is used like a ported estimator: it reads the features
# of one sample from its arguments or of several samples (one per line, the
# features separated by spaces) from the standard input:

# %%
from pathlib import Path

lines = [' '.join(map(str, sample)) for sample in X[::30].tolist()]
Path('/tmp/samples.txt').write_text('\n'.join(lines) + '\n')

print(Path('/tmp/samples.txt').read_text())

# %%
# !node /tmp/Pipeline.js < /tmp/samples.txt

# %% [markdown]
# ## Change the number of features
#
# The transformers can change the number of features (e.g. a
# `PolynomialFeatures` or a feature selector like `SelectKBest`); the ported
# pipeline reads the features of the pipeline. The class `Porter` creates a
# `Pipeline` for a pipeline:

# %%
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn_porter import Porter

pipe = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    SelectKBest(k=5),
    LogisticRegression(max_iter=1000),
)
pipe.fit(X, y)

porter = Porter(pipe, language='java')  # creates a `Pipeline`
y_classes, y_probas = porter.predict(X[::30])

print(y_classes)
print(y_probas)
print(porter.test(X[::15]))

# %% [markdown]
# ## Searches
#
# A pipeline as the best estimator of a `GridSearchCV` or a
# `RandomizedSearchCV` is ported too:

# %%
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()), {'svc__C': [0.1, 1.0, 10.0]}
)
search.fit(X, y)

print(test(search, X[::15], language='js'))

# %% [markdown]
# ## Unsupported preprocessing steps
#
# A pipeline with an unsupported transformer raises a `NotSupportedYetError`,
# because the predictions of the ported final estimator alone would differ
# from those of the pipeline. To port the final estimator anyway, pass
# `ignore_preprocessing=True` and apply the preprocessing steps to the input
# data of the ported estimator yourself, see the
# [limitations](https://github.com/nok/sklearn-porter/blob/main/docs/limitations.md).
