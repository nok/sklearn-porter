# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## GaussianNB
#
# Documentation: [sklearn.naive_bayes.GaussianNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

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
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X, y)

# %% [markdown]
# ### Transpile classifier

# %%
from sklearn_porter import Porter

porter = Porter(clf, language='java')
output = porter.export(export_data=True)

print(output)

# %% [markdown]
# ### Run classification in Java

# %%
# Save classifier:
# with open('GaussianNB.java', 'w') as f:
#     f.write(output)

# Check model data:
# $ cat data.json

# Download dependencies:
# $ wget -O gson.jar http://central.maven.org/maven2/com/google/code/gson/gson/2.8.5/gson-2.8.5.jar

# Compile model:
# $ javac -cp .:gson.jar GaussianNB.java

# Run classification:
# $ java -cp .:gson.jar GaussianNB data.json 1 2 3 4
