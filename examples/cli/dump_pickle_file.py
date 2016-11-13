from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn.externals import joblib

X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

joblib.dump(clf, 'model.pkl')

# Then execute the following command from the project root:
# python sklearn_porter/Porter.py -m examples/cli/model.pkl -o examples/cli/Model.java -l java
