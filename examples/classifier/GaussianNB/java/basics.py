from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
clf = GaussianNB()
clf.fit(X, y)

print(clf.predict([X[0]]))

# for i in dir(clf):
#     print(i, "  ", type(getattr(clf, i)))

# Cheese!

# result = Porter().port(clf)
# result = Porter(language='java').port(clf)
# print(result)

"""
"""