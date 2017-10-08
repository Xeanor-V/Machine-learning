import numpy as np, pylab, random, math

from cvxopt.solvers import qp
from cvxopt.base import matrix

## The larger the alpha value, the more important the point is to us
## for slack G int the second hald becomes 1
## for slack H becomes [0... ][C]
## if there's n
##Generate Test Data
classA = [(random.normalvariate(-1.5, 1),
           random.normalvariate(0.5, 1),
           1.0)
          for i in range(5)] + \
    [(random.normalvariate(1.5, 1),
      random.normalvariate(0.5, 1),
      1.0)
     for i in range(5)]

classB = [(random.normalvariate(10, 1),
           random.normalvariate(10, 1),
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
            p_matrix[i][j] = t[i]*t[j] * linear_kernel(np.array(data[i][0:2]),np.array(data[j][0:2]))
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
    sp_vectors = []
    epsilon = 0.00001
    for i in range(len(alpha_values)):
        if(math.fabs(alpha_values[i]) > epsilon):
            sp_vectors.append([data[i][0],data[i][1],data[i][2],alpha_values[i] ] )
    return np.array(sp_vectors)

#tells us which side of the decision boundary the point in on
def indicator(sp_vectors, vector):
    res = 0.0
    epsilon = 0.0000000001
    for i in range(len(sp_vectors)):
        ##print(sp_vectors[i][0:2])
        ##print(vector)
        res += sp_vectors[i][3]*sp_vectors[i][2]*linear_kernel(np.array(sp_vectors[i][0:2]),np.array(vector))
    return res
    #if res >0 :
    #    return 1
    #else:
    #    return -1
    

N = len(data)
P = build_P(data)
q = build_q(N)
G = build_G(N)
h = build_h(N)

r = qp(matrix(P) , matrix(q) , matrix(G) , matrix(h))
alpha_values = list(r['x'])
support_vectors = find_nonzero_values(alpha_values,data)
alpha = list(r['x'])

sp_vectors = build_nonzero(alpha,data)
print(sp_vectors)
##indicator = build_indicator(alpha,points)
##print(indicator)

## Plot test data
pylab.hold (True) 

pylab.plot([p[0] for p in classA], 
            [p[1] for p in classA],
            'bo')
pylab.plot([p[0] for p in classB], 
            [p[1] for p in classB], 
            'ro')


## Plot the decision boundary
xrange = np.arange(-10, 10, 0.05)
yrange = np.arange(-10, 10, 0.05)

grid = matrix([[ indicator(sp_vectors, [x,y])
                for y in yrange]
               for x in xrange])

pylab.contour(xrange, yrange, grid,
              (-1.0, 0.0, 1.0),
              colors=('red', 'black', 'blue'),
              linewidths=(1, 3, 1))
            
pylab.show()
