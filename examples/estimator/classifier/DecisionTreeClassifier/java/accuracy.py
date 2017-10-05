# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris

from sklearn_porter import Porter


iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
porter = Porter(clf)

accuracy = porter.predict_test([1.0, 2.0, 3.0, 4.0])
print(accuracy)  # 1.0

accuracy = porter.predict_test(X[0])
print(accuracy)  # 1.0

accuracy = porter.predict_test(X)
print(accuracy)  # 1.0
