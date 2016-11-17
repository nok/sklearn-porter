from sklearn import svm
from sklearn.datasets import load_iris

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
clf = svm.SVC(C=1., gamma=0.001, kernel='rbf', random_state=0)
clf.fit(X, y)

# Cheese!

result = Porter(language='c').port(clf)
print(result)
