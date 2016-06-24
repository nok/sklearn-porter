from sklearn.tree import tree
from sklearn.datasets import load_iris

from onl.nok.sklearn.export.Export import Export

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

# Cheese!

tree = Export.export(clf)
print(tree)