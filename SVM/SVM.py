import numpy as np
##from sklearn import svm

def Linear_Kernel(X, Y):
    return np.dot(X, Y.T) + 1

X = np.array([2,1])
Y = np.array([3,4])

print(Linear_Kernel(X,Y))