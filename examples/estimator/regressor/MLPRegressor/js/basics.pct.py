# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## MLPRegressor
#
# Documentation: [sklearn.neural_network.MLPRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

# %%
import sys
sys.path.append('../../../../..')

# %% [markdown]
# ### Load data

# %%
from sklearn.datasets import load_diabetes

samples = load_diabetes()
X = samples.data
y = samples.target

print(X.shape, y.shape)

# %% [markdown]
# ### Train regressor

# %%
from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(
    activation='relu', hidden_layer_sizes=30, max_iter=500, alpha=1e-4,
    solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)
reg.fit(X, y)

# %% [markdown]
# ### Transpile regressor

# %%
from sklearn_porter import Porter

porter = Porter(reg, language='js')
output = porter.export()

print(output)

# %% [markdown]
# ### Run regression in JavaScript

# %%
# Save regressor:
# with open('MLPRegressor.js', 'w') as f:
#     f.write(output)

# Run regression:
# if hash node 2/dev/null; then
#     node MLPRegressor.js 0.03 0.05 0.06 0.02 -0.04 -0.03 -0.04 -0.002 0.01 -0.01
# fi
