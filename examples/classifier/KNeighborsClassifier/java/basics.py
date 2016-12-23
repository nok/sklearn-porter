from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
clf = KNeighborsClassifier(algorithm='brute', n_neighbors=1, weights='distance')
clf.fit(X, y)

# Cheese!

result = Porter().port(clf)
# model = Porter(language='java').port(clf)
print(result)
