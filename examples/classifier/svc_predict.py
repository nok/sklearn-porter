from sklearn import svm
from sklearn.datasets import load_iris

from onl.nok.sklearn.Porter import port

import numpy as np
import math

iris = load_iris()
clf = svm.SVC(C=1., gamma=0.001, kernel='rbf', random_state=0)
clf.fit(iris.data, iris.target)

print(type(clf))

# Cheese!

# print(clf.get_params())

# poly:
# k (x, y) = (gamma * x*y = coef0) ^ degree

# print('gamma', clf.gamma)
# print('coef', clf.coef0)
# print('degree', clf.degree)

# rbf:
# k (x, y) = exp( -gamma * || x-y || ^ 2 )
# if gamma > 0


# I've only implemented the linear and rbf kernels
def kernel(params, sv, X):
    if params['kernel'] == 'linear':
        gh = []
        for vi in sv:
            gh.append(np.dot(vi, X))
        return gh
    elif params['kernel'] == 'rbf':
        gh = []
        for vi in sv:
            # print vi, X, np.dot(vi - X, vi - X)
            # exit()
            r = np.math.exp(-params['gamma'] * np.dot(vi - X, vi - X))
            gh.append(r)
        return gh


# This replicates clf.decision_function(X)
def decision_function(params, sv, nv, a, b, X):
    # calculate the kernels
    k = kernel(params, sv, X)

    # print k

    # define the start and end index for support vectors for each class
    start = [sum(nv[:i]) for i in range(len(nv))]
    # print start

    end = [start[i] + nv[i] for i in range(len(nv))]
    # print end

    # calculate: sum(a_p * k(x_p, x)) between every 2 classes
    c = [sum(a[i][p] * k[p] for p in range(start[j], end[j])) + sum(a[j-1][p] * k[p] for p in range(start[i], end[i])) for i in range(len(nv)) for j in range(i+1, len(nv))]

    # add the intercept
    return [sum(x) for x in zip(c, b)]


# This replicates clf.predict(X)
def predict(params, sv, nv, a, b, cs, X):
    ''' params = model parameters
        sv = support vectors
        nv = # of support vectors per class
        a  = dual coefficients
        b  = intercepts
        cs = list of class names
        X  = feature to predict
    '''
    decision = decision_function(params, sv, nv, a, b, X)
    votes = [(i if decision[p] > 0 else j) for p,(i,j) in enumerate((i,j) for i in range(len(cs)) for j in range(i+1,len(cs)))]
    # print('votes', votes)

    return cs[max(set(votes), key=votes.count)]

# ------------------------------------------------------------

# Compare with the builtin predict
print(clf.predict(np.array([[-9., 2., 0.5, 1.]])))

X = np.array([-9., 2., 0.5, 1.])

# Get parameters from model
params = clf.get_params()
print(params)

sv = clf.support_vectors_
nv = clf.n_support_
a = clf.dual_coef_
b = clf._intercept_
cs = clf.classes_

# Use the functions to predict
print(predict(params, sv, nv, a, b, cs, X))

# ------------------------------------------------------------

# Get parameters from model
params = clf.get_params()

sv = clf.support_vectors_  # sv = support vectors
nv = clf.n_support_  # n support vectors per class
a = clf.dual_coef_  # dual coefficients
b = clf._intercept_  # intercepts
cs = clf.classes_  # list of class names
X = np.array([-9., 2., 0.5, 1.])  # features

print ("------------------------------------")

# calculate the kernels
kernels = []
if params['kernel'] is 'rbf':
    # print(sv[0])
    # print(nv)
    # exit()
    for vectors in sv:
        o_k = 0.0
        for idx in range(len(vectors)):
            delta = vectors[idx] - X[idx]
            o_k += delta * delta
        o_k = math.exp(-params['gamma'] * o_k)
        kernels.append(o_k)
# if params['kernel'] is 'linear':
#     for vectors in sv:
#         o_k = 0.0
#         for idx in range(len(vectors)):
#             o_k += vectors[idx] * X[idx]
#         kernels.append(o_k)

# print(kernels)

# define the start and end index for support vectors for each class
starts = []
for i in range(len(nv)):
    if i != 0:
        start = 0
        for j in range(i):
            start += nv[j]
        starts.append(start)
    else:
        starts.append(0)
print(starts)
ends = []
for i in range(len(nv)):
    ends.append(nv[i] + starts[i])
print(ends)

# TODO: Continue here:

# calculate: sum(a_p * k(x_p, x)) between every 2 classes
c = []
for i in range(len(nv)):
    for j in range(i + 1, len(nv)):
        o_a = 0.0
        for p in range(starts[j], ends[j]):
            o_a += a[i][p] * kernels[p]
        o_b = 0.0
        for p in range(starts[i], ends[i]):
            o_b += a[j - 1][p] * kernels[p]
        c.append(o_a + o_b)

# add the intercept
for i in range(len(c)):
    c[i] += b[i]

votes = []
for i in range(len(cs)):
    for j in range(i + 1, len(cs)):
        if c[i] > 0:
            votes.append(i)
        else:
            votes.append(j)
        # tmp.append([i, j])
        # votess.append(tmp[i][0] if c[i] > 0 else tmp[i][1])

classes = {}
for i in range(len(votes)):
    if votes[i] in classes:
        classes[votes[i]] += 1
    else:
        classes[votes[i]] = 1

print('classes', classes)

class_idx = -1
counter = -1
for count in classes:
    if classes[count] > counter:
        counter = classes[count]
        class_idx = count

print(cs[class_idx])

print ("------------------------------------")

print(port(clf))
