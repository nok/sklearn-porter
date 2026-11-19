# %% [markdown]
# # Pipelines
#
# This example ports a `Pipeline` of scikit-learn to JavaScript with
# [sklearn-porter](https://github.com/nok/sklearn-porter): the preprocessing
# and the classifier in one source file, which predicts the classes of the
# raw features like the pipeline.
#
# Running the ported pipeline requires `node`.
#
# ## Why the preprocessing matters
#
# The features of the wine dataset have very different scales. A support
# vector machine without scaling predicts much worse than with a
# `StandardScaler` before it:

# %%
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, stratify=y
)

svc = SVC().fit(X_train, y_train)
pipe = make_pipeline(StandardScaler(), SVC()).fit(X_train, y_train)

print(f'SVC alone:          {svc.score(X_test, y_test):.2f}')
print(f'StandardScaler+SVC: {pipe.score(X_test, y_test):.2f}')

# %% [markdown]
# So porting only the classifier isn't enough. sklearn-porter ports the
# whole pipeline (a pipeline with a transformer which can't be ported raises
# an error).
#
# ## Port the pipeline
#
# The functions `port`, `save`, `make` and `test` (and the class `Porter`)
# accept pipelines like classifiers. Save the ported pipeline:

# %%
from sklearn_porter import save

src_path = save(pipe, language='js', directory='/tmp')

print(src_path)

# %% [markdown]
# ## Run the ported pipeline
#
# The ported pipeline reads the raw features and scales them before the
# prediction:

# %%
# !node /tmp/Pipeline.js 14.23 1.71 2.43 15.6 127 2.8 3.06 0.28 2.29 5.64 1.04 3.92 1065

# %% [markdown]
# It also reads several samples (one per line) from the standard input:

# %%
from pathlib import Path

lines = [' '.join(map(str, sample)) for sample in X_test[:3].tolist()]
Path('/tmp/samples.txt').write_text('\n'.join(lines) + '\n')

print(Path('/tmp/samples.txt').read_text())

# %%
# !node /tmp/Pipeline.js < /tmp/samples.txt

# %% [markdown]
# ## Use the pipeline in your own code
#
# The ported pipeline is also a class for your own code. Its `predict`
# scales the raw features before the prediction like the pipeline (see the
# [usage](https://github.com/nok/sklearn-porter/blob/main/docs/usage.md) of
# the other languages):

# %%
# !node -e "const Pipeline = require('/tmp/Pipeline.js'); console.log('Class: ' + new Pipeline().predict([14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]))"

# %% [markdown]
# ## Check the ported pipeline
#
# `make` returns the predictions of the ported pipeline, `test` compares
# them with those of the pipeline:

# %%
from sklearn_porter import make, test

y_pred, _ = make(pipe, X_test[:5], language='js')

print(y_pred)
print(pipe.predict(X_test[:5]))
print(test(pipe, X_test, language='js'))

# %% [markdown]
# ## Change the number of features
#
# The transformers can change the number of features, e.g. a
# `PolynomialFeatures` (more features) and a feature selector like
# `SelectKBest` (fewer features):

# %%
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

pipe = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    SelectKBest(k=10),
    LogisticRegression(max_iter=1000),
)
pipe.fit(X_train, y_train)

print(test(pipe, X_test, language='js'))

# %% [markdown]
# ## Searches
#
# The best pipeline of a `GridSearchCV` or a `RandomizedSearchCV` is ported
# too:

# %%
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()), {'svc__C': [0.1, 1.0, 10.0]}
)
search.fit(X_train, y_train)

print(test(search, X_test, language='js'))

# %% [markdown]
# A pipeline with a transformer which can't be ported raises a
# `NotSupportedYetError`, because the ported classifier alone would predict
# other classes. See the
# [limitations](https://github.com/nok/sklearn-porter/blob/main/docs/limitations.md)
# for the workaround with `ignore_preprocessing=True`.
