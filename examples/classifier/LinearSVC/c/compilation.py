# -*- coding: utf-8 -*-

import subprocess as subp

from sklearn import svm
from sklearn.datasets import load_iris

from sklearn_porter import Porter


iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

# Cheese!

data = Porter(clf, language='c').export(details=True)

# Save model:
with open(data.get('filename'), 'w') as f:
    f.write(data.get('model'))

# Compile model:
command = data.get('cmd').get('compilation')
subp.call(command, shell=True)

# Use the model:
features = ' '.join([repr(x) for x in X[0]])
command = '%s %s' % (data.get('cmd').get('execution'), features)
prediction = subp.check_output(command, shell=True)

print('Ported classifier: %s' % prediction)  # class: 0
print('Original classifier: %s' % clf.predict([X[0]])[0])  # class: 0
