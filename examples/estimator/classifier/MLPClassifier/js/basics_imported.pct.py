# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## MLPClassifier
#
# Documentation: [sklearn.neural_network.MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

# %%
import sys
sys.path.append('../../../../..')

# %% [markdown]
# ### Load data

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X = shuffle(X, random_state=0)
y = shuffle(y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=5)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %% [markdown]
# ### Train classifier

# %%
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='relu', hidden_layer_sizes=50,
                    max_iter=500, alpha=1e-4, solver='sgd',
                    tol=1e-4, random_state=1, learning_rate_init=.1)
clf.fit(X_train, y_train)

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
# with open('MLPClassifier.js', 'w') as f:
#     f.write(output)

# Check model data:
# $ cat data.json

# Run classification:
# if hash node 2/dev/null; then
#     python -m SimpleHTTPServer 8877 & serve_pid=$!
#     node MLPClassifier.js http://127.0.0.1:8877/data.json 1 2 3 4
#     kill $serve_pid
# fi
