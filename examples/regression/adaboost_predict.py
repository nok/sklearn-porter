from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# from onl.nok.sklearn.export.Export import Export

iris = load_iris()
base_estimator = DecisionTreeClassifier(max_depth=4)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100)
clf.fit(iris.data, iris.target)

# Cheese!

# TODO: Implement and add export
# trees = Export.export(clf)
# print(trees)