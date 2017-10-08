import numpy as np, pylab, random, math

from cvxopt.solvers import qp
from cvxopt.base import matrix

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

class_info = [x[2] for x in data]
print(class_info)
