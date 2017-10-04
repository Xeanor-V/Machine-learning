import numpy as np, pylab, random, math
from cvxopt.solvers import qp 
from cvxopt.base import matrix

##from sklearn import svm
## P matrix, t
## equation 6
def Linear_Kernel(X, Y):
    return np.dot(X, Y.T) + 1

def Build_P(X,t):
    P_Matrix = [[0 for x in range(len(X))] for y in range(len(X))] 
    for i in range(len(X)):
        for j in range(len(X)):
            P_Matrix[i][j] = t[i]*t[j] * Linear_Kernel(X[i],X[j])
    return np.array(P_Matrix, dtype = np.dtype('d'))

def Build_q(N):
    return np.ones(N) * -1

def Build_h(N):
    return np.zeros(N)

def Build_G(N):
    G_Matrix = [[0 for x in range(N)] for y in range(N)] 
    for i in range(N):
        for j in range(N):
            if(i==j):
                G_Matrix[i][j] = -1
    return np.array(G_Matrix, dtype = np.dtype('d'))

def Build_NonZero(Alpha_Values,X,Y):
    Alpha = []
    Points = []
    epsilon = 0.00001
    for i in range(len(Alpha_Values)):
        if(math.fabs(Alpha_Values[i]) > epsilon):
            Alpha.append(Alpha_Values[i])
            Points.append([X[i], Y[i]])
    return np.array(Alpha), np.array(Points)

def Build_Indicator(Alpha,Points,T):
    res = 0.0
    for i in range(len(Alpha)):
        res += Alpha[i]*T[i]*Linear_Kernel(Points,Points[i])
    return res

    

T = np.array([1,-1])
X = np.array([2,1])
Y = np.array([3,4])
N = len(X)
q = Build_q(N)
P = Build_P(X,T)
G = Build_G(N)
h = Build_h(N)

##print(len(Q))
print(Linear_Kernel(X,Y))


r = qp( matrix(P) , matrix(q) , matrix(G) , matrix(h)) 
alpha = list ( r [ 'x' ])

Alpha,Points = Build_NonZero(alpha,X,Y)
indicator = Build_Indicator(Alpha,Points,T)
print(indicator)


