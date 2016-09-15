from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn.externals import joblib

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

joblib.dump(clf, 'model.pkl')

# Then execute the following command from the project root:
# python onl/nok/sklearn/Porter.py -m examples/cli/model.pkl -o examples/cli/Model.java -l java
