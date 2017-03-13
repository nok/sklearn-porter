# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris

from sklearn_porter import Porter


iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Cheese!

porter = Porter(clf)

# Single prediction:
y = porter.predict(X[0])
print(y)
# 0

# Multiples predictions:
Y = porter.predict(X)
print(Y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
