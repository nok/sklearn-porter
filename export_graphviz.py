import sys
import subprocess
from sklearn.datasets import load_iris
from sklearn import tree


def main():
    clf = tree.DecisionTreeClassifier()
    iris = load_iris()

    clf = clf.fit(iris.data, iris.target)

    filename = 'tree'
    tree.export_graphviz(clf, out_file=filename + '.dot')
    subprocess.call('dot -Tpng %s -o %s' % (filename + '.dot', filename + '.png'), shell=True)


if __name__ == "__main__":
    main()