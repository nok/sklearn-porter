# %% [markdown]
# # sklearn-porter
#
# Transpile trained scikit-learn estimators to C, Java, JavaScript and others.
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## Basics
#
# **Step 1**: Load data and train a dummy classifier:

# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)

# %% [markdown]
# **Step 2**: Port or transpile an estimator:

# %%
from sklearn_porter import port, save, make, test

output = port(clf, language='js', template='attached')

print(output)

# %% [markdown]
# **Step 3**: Save the ported estimator:

# %%
src_path, json_path = save(clf, language='js', template='exported', directory='/tmp')

print(src_path, json_path)

# %% [shell]
# cat /tmp/DecisionTreeClassifier.js | pygmentize -l javascript

# %% [shell]
# cat /tmp/DecisionTreeClassifier.json | pygmentize -l json

# %% [markdown]
# **Step 4**: Make predictions with the ported estimator:

# %%
y_classes, y_probas = make(clf, X[:10], language='js', template='exported')

print(y_classes, y_probas)

# %% [markdown]
# **Step 5**: Test always the ported estimator by making an integrity check:

# %%
score = test(clf, X[:10], language='js', template='exported')

print(score)

# %% [markdown]
# ## OOP
#
# **Step 1**: Port or transpile an estimator:

# %%
from sklearn_porter import Estimator

est = Estimator(clf, language='java', template='attached')
output = est.port()

print(output)

# %% [markdown]
# **Step 2**: Save the ported estimator:

# %%
est.template = 'exported'
src_path, json_path = est.save(directory='/tmp')

print(src_path, json_path)

# %% [shell]
# cat /tmp/DecisionTreeClassifier.java | pygmentize -l java

# %% [shell]
# cat /tmp/DecisionTreeClassifier.json | pygmentize -l json

# %% [markdown]
# **Step 3**: Make predictions with the ported estimator:

# %%
y_classes, y_probas = est.make(X[:10])

print(y_classes, y_probas)

# %% [markdown]
# **Step 4**: Test always the ported estimator by making an integrity check:

# %%
score = est.test(X[:10])

print(score)
