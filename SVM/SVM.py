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

def Build_NonZero(Alpha_Values,data):
    Alpha = []
    Points = []
    epsilon = 0.00001
    for i in range(len(Alpha_Values)):
        if(math.fabs(Alpha_Values[i]) > epsilon):
            Points.append(data[i])
    return np.array(Points)

#tells us which side of the decision boundary the point in on
def Build_Indicator(Alpha,Points,T):
    res = 0.0
    for i in range(len(Alpha)):
        res += Alpha[i]*T[i]*Linear_Kernel(Points,Points[i])
    return res

N = len(data)
P = Build_P(data)
q = Build_q(N)
G = Build_G(N)
h = Build_h(N)

##print(len(Q))
##print(Linear_Kernel(X,Y))

r = qp(matrix(P) , matrix(q) , matrix(G) , matrix(h))
alpha = list(r['x'])

Alpha,Points = Build_NonZero(alpha,X,Y)
indicator = Build_Indicator(Alpha,Points,T)
##print(indicator)

##Plot test data
pylab.hold (True) 
pylab.plot([p[0] for p in classA], 
            [p[1] for p in classA],
            'bo')
pylab.plot([p[0] for p in classA], 
            [p[1] for p in classB], 
            'ro')
pylab.show()

## Plotting the Decision Boundary
#values go from -4 to 4 with a step of 0.05
xrange = np.arange(-4, 4, 0.05)
yrange = np.arange(-4, 4, 0.05)

grid = matrix([[indicator(x,y)
                for y in yrange]
               for x in xrange])

pylab.contour(xrange, yrange, grid,
              (-1.0, 0.0, 1.0),
              colors=('red', 'black', 'blue'),
              linewidths=(1, 3, 1))


