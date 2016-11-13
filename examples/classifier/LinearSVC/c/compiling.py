import subprocess as subp

from sklearn import svm
from sklearn.datasets import load_iris

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

# Cheese!

data = Porter(language='c', with_details=True).port(clf)

# Save model:
with open(data.get('filename'), 'w') as file:
    file.write(data.get('model'))

# Compile model:
command = data.get('compiling_cmd')
subp.call(command, shell=True)

# Use the model:
features = ' '.join([repr(x) for x in X[0]])
command = '%s %s' % (data.get('execution_cmd'), features)
prediction = subp.check_output(command, shell=True)

print('Ported classifier: %s' % prediction)  # class: 0
print('Original classifier: %s' % clf.predict([X[0]])[0])  # class: 0
