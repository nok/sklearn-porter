# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)

# %% [markdown]
# 1. Load data and train a dummy classifier:

# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)

# %% [markdown]
# 2. Port or transpile an estimator:

# %%
from sklearn_porter import port, save, make, test

output = port(clf, language='js', template='attached')
print(output)

# %% [markdown]
# 3. Save the ported estimator:

# %%
src_path, json_path = save(
    clf, language='js', template='exported', directory='/tmp'
)
print(src_path, json_path)

# %% [markdown]
# 4. Make predictions with the ported estimator:

# %%
y_classes, y_probas = make(clf, X[:10], language='js', template='exported')
print(y_classes, y_probas)

# %% [markdown]
# 5. Test always the ported estimator by making an integrity check:

# %%
score = test(clf, X[:10], language='js', template='exported')
print(score)