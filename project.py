import numpy as np
from numpy import *
import numdifftools as nd
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
import inspect
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

    def inverse(self, x_n):
        try:
            return inv(self.hessian(x_n)) if len(x_n)!=1 else 1/self.hessian(x_n)[0]
        except:
            return inv(self.hessian(x_n) + np.identity(x_n.size)) if len(x_n)!=1 else 1/self.hessian(x_n)[0]

    def square_lambda(self, x_n):
        return self.gradient(x_n).T.dot(self.inverse(x_n).dot(self.gradient(x_n)))
    
    def line_searching(self, x_n, t):
        i = 0
        a = 1/3; b = 1/2
        lmbd = self.square_lambda(x_n)
        delta =  -self.inverse(x_n).dot(self.gradient(x_n))
        while not self.function(x_n + t * delta) < self.function(x_n) + a *t * lmbd and i < 100:
            t *= b
            i += 1
        return t
    
    def method(self):
        lmbd = self.square_lambda(self.x_n)
        while not 0.5 * lmbd <= self.e and self.iteration < 1000:
            self.t = self.line_searching(self.x_n, self.t)
            delta = -self.inverse(self.x_n).dot(self.gradient(self.x_n))
            self.x_n = self.x_n + self.t * delta
            lmbd = self.square_lambda(self.x_n)
            self.iteration += 1
            print(f"{self.iteration})  t = {self.t}  x = {self.x_n}  f(x) = {self.function(self.x_n)} ")
            self.t = self.init_t
            self.x.append(self.x_n)
        self.x = np.array(self.x)

if __name__=="__main__":
    Newton_method(
        function = lambda x: sin(x[0]) + x[0]**4, 
        x_n = np.array([1])
    )