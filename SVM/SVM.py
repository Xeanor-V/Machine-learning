import numpy as np, pylab, random, math

from cvxopt.solvers import qp
from cvxopt.base import matrix

##The larger the alpha value, the more important the point is to us

##Generate Test Data
classA = [(random.normalvariate(-1.5, 1),
           random.normalvariate(0.5, 1),
           1.0)
          for i in range(5)] + \
    [(random.normalvariate(1.5, 1),
      random.normalvariate(0.5, 1),
      1.0)
     for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5),
           random.normalvariate(-0.5, 0.5),
           -1.0)
          for i in range(10)]

data = classA + classB
random.shuffle(data)

#Kernel Functions
def linear_kernel(X, Y):
    return np.dot(X, Y.T) + 1

def polynomial_kernel(X,Y,p):
    return linear_kernel(X,Y)**p

#Building matrix / vector functions
def build_P(data):
    t = [x[2] for x in data]
    p_matrix = [[0 for x in range(len(data))] for y in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data)):
            p_matrix[i][j] = t[i]*t[j] * linear_kernel(np.array(data[i][0:1]),np.array(data[j][0:1]))
    return np.array(p_matrix, dtype = np.dtype('d'))

def build_q(N):
    return np.ones(N) * -1

def build_h(N):
    return np.zeros(N)

def build_G(N):
    g_matrix = [[0 for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(N):
            if(i==j):
                g_matrix[i][j] = -1
    return np.array(g_matrix, dtype = np.dtype('d'))

def build_nonzero(alpha_values,data):
    alpha = []
    points = []
    epsilon = 0.00001
    for i in range(len(alpha_values)):
        if(math.fabs(alpha_values[i]) > epsilon):
            points.append(data[i])
    return np.array(points)

#tells us which side of the decision boundary the point in on
def build_indicator(alpha,points):
    res = 0.0
    for i in range(len(alpha)):
        res += alpha[i]*T[i]*linear_kernel(points,points[i])
    return res

N = len(data)
P = build_P(data)
q = build_q(N)
G = build_G(N)
h = build_h(N)

r = qp(matrix(P) , matrix(q) , matrix(G) , matrix(h))
alpha = list(r['x'])

points = build_nonzero(alpha,data)
indicator = build_indicator(alpha,points)
##print(indicator)

## Plot test data
pylab.hold (True) 
pylab.plot([p[0] for p in classA], 
            [p[1] for p in classA],
            'bo')
pylab.plot([p[0] for p in classA], 
            [p[1] for p in classB], 
            'ro')
pylab.show()

## Plot the decision boundary
xrange = np.arange(-4, 4, 0.05)
yrange = np.arange(-4, 4, 0.05)

grid = matrix([[indicator(x,y)
                for y in yrange]
               for x in xrange])

pylab.contour(xrange, yrange, grid,
              (-1.0, 0.0, 1.0),
              colors=('red', 'black', 'blue'),
              linewidths=(1, 3, 1))


