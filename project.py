import numpy as np
from numpy import *
import numdifftools as nd
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
import pandas as pd
import inspect
import warnings
warnings.filterwarnings("ignore")

class Newton_method:
    def __init__(self, function, x_n, t = 1, e = 0.01):
        self.function = function
        self.x_n = x_n
        self.init_t = t
        self.t = t
        self.iteration = 0
        self.e = e
        self.x = [self.x_n]
        self.__str__()

    gradient = lambda self, x_n: nd.Gradient(self.function)(x_n)

    hessian = lambda self, x_n: nd.Hessian(self.function)(x_n)

    def inverse(self, x_n):
        try:
            return inv(self.hessian(x_n)) if len(x_n)!=1 else 1/self.hessian(x_n)[0]
        except:
            return inv(self.hessian(x_n) + np.identity(x_n.size)) if len(x_n)!=1 else 1/self.hessian(x_n)[0]

    def square_lambda(self, x_n):
        return self.gradient(x_n).T.dot(self.inverse(x_n).dot(self.gradient(x_n)))
    
    def line_searching(self, x_n, lmbd, delta, t):
        i = 0
        a = 1/3; b = 1/2
        # lmbd = self.square_lambda(x_n)
        # delta =  -self.inverse(x_n).dot(self.gradient(x_n))
        while not self.function(x_n + t * delta)\
              < self.function(x_n) + a *t * lmbd and i < 100:
            t *= b
            i += 1
        return t
    
    def method(self):
        
        while not 0.5 * (lmbd:=self.square_lambda(self.x_n)) <= self.e\
              and self.iteration < 1000:
            
            delta = -self.inverse(self.x_n).dot(self.gradient(self.x_n))
            self.t = self.line_searching(self.x_n, lmbd, delta, self.t)

            self.x_n = self.x_n + self.t * delta
            # lmbd = self.square_lambda(self.x_n)
            
            self.iteration += 1
            # print(f"{self.iteration})  t = {self.t}  x = {self.x_n}  f(x) = {self.function(self.x_n)} ")
            
            self.t = self.init_t
            self.x.append(self.x_n)

        self.x = np.array(self.x)
    
    def __str__(self):

        self.method()

        print(f" After {self.iteration} iterations:")
        print(f"Solution is: x* = {self.x_n} p* = {self.function(self.x_n)}")

    def test(self, exact_x):
        df = pd.DataFrame({
            "exact": exact_x,
            "calculated": self.x_n,
            "abs error": np.max(abs(self.x_n - np.array(exact_x))),
        })
        print('\nVerification:')
        print(df)

    def plot(self):
        if self.x_n.size ==1:
            fig, ax = plt.subplots(figsize=(8, 6))
            title = '$\ f(x)=' + 'x'.join(
            ''.join(
                '^'.join(
                    ''.join(
                        ''.join(
                            inspect.getsource(self.function).split(': ')[1].split(',')[0].split('[')
                            ).split(']')
                        ).split('**')
                    ).split('*')
                ).split('x0')) + '$'
            ax.grid()
            ax.plot(np.linspace(-3,3,100), self.function([np.linspace(-3,3,100)]))
            # ax.plot(self.x[-1], self.function(self.x[-1]), 'o')
            ax.scatter(self.x, list(map(lambda x: self.function(x), self.x)))
            # ax.scatter(se)
            for i in 'xy':
                exec(f"ax.set_{i}label('$\ {i.upper()}  axis$')")
            ax.set_title(title, fontsize=18)
            fig.show()
        elif self.x_n.size == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            title = '$\ f(x)=' + ''.join(
                '^'.join(
                    ''.join(
                        '_'.join(
                            inspect.getsource(self.function).split(': ')[1].split(',')[0].split('[')
                            ).split(']')
                        ).split('**')
                    ).split('*')
                ) + '$'

            ax = plt.axes(projection = '3d')
            ax.grid()
            a = b = np.linspace(-3,3,100)
            a, b = np.meshgrid(a, b)
            f = self.function([a, b])
            ax.plot_surface(a, b, f, cmap='viridis',edgecolor='green')
            ax.plot_surface(
                self.x.T[0], self.x.T[1],
                self.function(np.meshgrid(self.x[:,0], self.x[:,1])),
                cmap = 'binary', edgecolor = 'red'
            )
            for i in 'xyz':
                exec(f"ax.set_{i}label('$\ {i.upper()}  axis$')")
            ax.set_title(title, fontsize=18)
            ax.view_init(30, 35)
            fig.show()

if __name__=="__main__":
    # solution = Newton_method(
    #     function = lambda x: sin(x[0]) + x[0]**4, 
    #     x_n = np.array([1])
    # )
    # solution.plot()
    # plt.show()

    # solution = Newton_method(
    #     function = lambda x: (x[0] + 2)**2 + 3*(x[1] - 1)**2 + 1,
    #     x_n = np.array([2, 2])
    # )
    # solution.plot()
    # plt.show()

    solution = Newton_method(
        function = lambda x: x[0]**2 + x[1]**2 + 2*x[1] + x[2]**2 + (x[3] + 4)**2 + (x[4] - 1)**2 + (x[5] - 3)**2,
        x_n = np.zeros(6)
    )
    8

    # verification

    # solution = Newton_method(
    #     function = lambda x: x[0]**2 - 4, 
    #     x_n = np.array([3])
    # )
    # solution.test(exact_x=[0])
    # solution.plot()
    # plt.show()

    # solution = Newton_method(
    #     function = lambda x: x[0]**2 - 3*x[0], 
    #     x_n = np.array([3])
    # )
    # solution.test(exact_x=[1.5])
    # solution.plot()
    # plt.show()