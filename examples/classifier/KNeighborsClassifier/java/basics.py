from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
clf = KNeighborsClassifier(algorithm='brute', n_neighbors=1, weights='distance')
clf.fit(X, y)

# print(clf.predict([[5.4231313, 3.712313223, 1.5123123, 0.2123123]]))

# Cheese!

result = Porter().port(clf)
# model = Porter(language='java').port(clf)
# print(result)
