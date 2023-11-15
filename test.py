import unittest
from project import Newton_method
import numpy as np

class TestNewtonMethod(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-2  # Tolerance level for floating point comparison

    def assertArrayAlmostEqual(self, calculated_solution, expected_solution, tol):
        
        assert abs(calculated_solution - expected_solution) <= tol

    def test_case_1(self):
        # Simple quadratic function: f(x) = (x - 5)^2
        def f(x): return (x[0] - 5)**2
        initial_guess = np.array([0])
        solver = Newton_method(function=f, x_n=initial_guess)
        expected_solution = np.array([5])
        self.assertArrayAlmostEqual(f(solver.x_n), f(expected_solution), self.tolerance)
        print("The first testcase passed succesfully!")

    def test_case_2(self):
        # Test case for 2D function: Rosenbrock function, minimum at [1, 1]
        def f(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        initial_guess = np.array([-1.2, 1])
        solver = Newton_method(function=f, x_n=initial_guess)
        expected_solution = np.array([1, 1])
        self.assertArrayAlmostEqual(f(solver.x_n), f(expected_solution), self.tolerance)
        print("The second testcase passed succesfully!")

    def test_case_3(self):
        # Test case for 3D function: f(x, y, z) = x^2 + y^2 + (z-1)^2, minimum at [0, 0, 1]
        def f(x): return x[0]**2 + x[1]**2 + (x[2] - 1)**2
        initial_guess = np.array([0.5, -0.5, 0.5])
        solver = Newton_method(function=f, x_n=initial_guess)
        expected_solution = np.array([0, 0, 1])
        self.assertArrayAlmostEqual(f(solver.x_n), f(expected_solution), self.tolerance)
        print("The third testcase passed succesfully!")

if __name__ == '__main__':
    unittest.main()
