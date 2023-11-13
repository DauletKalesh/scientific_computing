import numpy as np
from numpy import *
import numdifftools as nd
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
import inspect
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

class Newton_method:
    def __init__(self, function, x_n, t = 1, e = 0.01):
        self.function = function
        self.x_n = x_n
        self.init_t = t
        self.t = t
        self.e = e
        self.x = [self.x_n]
    gradient = lambda self, x_n: nd.Gradient(self.function)(x_n)

    hessian = lambda self, x_n: nd.Hessian(self.function)(x_n)


if __name__=="__main__":
    Newton_method(
        function = lambda x: sin(x[0]) + x[0]**4, 
        x_n = np.array([1])
    )