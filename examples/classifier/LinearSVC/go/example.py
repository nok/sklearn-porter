from sklearn import svm
from sklearn.datasets import load_iris

from onl.nok.sklearn.Porter import port

iris = load_iris()
clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(iris.data, iris.target)

# Cheese!

print(port(clf, language='go'))
