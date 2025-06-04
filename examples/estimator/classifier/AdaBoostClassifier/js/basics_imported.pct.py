# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## AdaBoostClassifier
#
# Documentation: [sklearn.ensemble.AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

base_estimator = DecisionTreeClassifier(max_depth=4, random_state=0)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100,
                         random_state=0)
clf.fit(X, y)

# %% [markdown]
# ### Transpile classifier

# %%
from sklearn_porter import Porter

porter = Porter(clf, language='js')
output = porter.export(export_data=True)

print(output)

# %% [markdown]
# ### Run classification in JavaScript

# %%
# Save classifier:
# with open('AdaBoostClassifier.js', 'w') as f:
#     f.write(output)

# Check model data:
# $ cat data.json

# Run classification:
# if hash node 2/dev/null; then
#     python -m SimpleHTTPServer 8877 & serve_pid=$!
#     node AdaBoostClassifier.js http://127.0.0.1:8877/data.json 1 2 3 4
#     kill $serve_pid
# fi
