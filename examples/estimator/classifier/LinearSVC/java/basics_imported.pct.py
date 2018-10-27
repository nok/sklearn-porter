# %% [markdown]
# # sklearn-porter
#
# Repository: [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter)
#
# ## LinearSVC
#
# Documentation: [sklearn.svm.LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

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
from sklearn import svm

clf = svm.LinearSVC(C=1., random_state=0)
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
# with open('LinearSVC.java', 'w') as f:
#     f.write(output)

# Check model data:
# $ cat data.json

# Download dependencies:
# $ wget -O gson.jar http://central.maven.org/maven2/com/google/code/gson/gson/2.8.5/gson-2.8.5.jar

# Compile model:
# $ javac -cp .:gson.jar LinearSVC.java

# Run classification:
# $ java -cp .:gson.jar LinearSVC data.json 1 2 3 4
