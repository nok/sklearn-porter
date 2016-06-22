from sklearn.tree import tree
from sklearn.datasets import load_iris

from nok.Export import Export

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

# Cheese!
tree = Export.predict(clf)
print(tree)