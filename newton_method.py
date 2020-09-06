import math
from decimal import Decimal # TODO : USE DECIMAL!!!
import random
from typing import Optional, Union
from matplotlib import pyplot as plt

import numpy as np

def count_gradient(function, variables_point : Union[list, tuple], variables_delta : Union[float, tuple, list]):
    if type(variables_delta) == float:
        temp = variables_delta
        variables_delta = [temp for _ in range(len(variables_point))]
    res = []
    f_x0 = function(*variables_point)
    for index, var in enumerate(variables_point):
        this_new_val = variables_delta[index] + var
        this_new_vars = tuple(list(variables_point[:index]) + [this_new_val] + list(variables_point[index + 1:]))

        this_new_func = function(*this_new_vars)

        res.append((this_new_func - f_x0) / (this_new_val - var))

    return res

def gradient_optimize(function, point : tuple, diff_delta : float, learning_rate : float, iterations : int, debug = False):
    solution : list = list(point)
    for iteration in range(iterations):
        this_grad = count_gradient(function, tuple(solution), tuple([diff_delta for _ in range(len(point))]))
        this_alpha = learning_rate * (1 - iteration / iterations) ** (1 / 3) # (math.sqrt())
        this_increment = [_i * this_alpha for _i in this_grad]

        solution = [solution[_i] - this_increment[_i] for _i in range(len(solution))]

        if debug and random.random() < (5 / iterations):
            print(this_grad, this_alpha, this_increment, solution)

    return function(*solution), solution

def find_newton_root(function, point : tuple, iterations : int, debug = False):
    solution  = list(point)
    err_arr = []
    for iteration in range(iterations):
        df = count_gradient(function, tuple(solution), 0.000001)
        f = function(*solution)

        # print(f, df)
        # solution -= f / df
        if debug:
            print("Point:", solution, "F:", f, "dF:", df)
        try:
            minus = [0.1 * f / df[i_dim] for i_dim in range(len(df))]
        except Exception as e:
            print("Point:", solution, "F:", f, "dF:", df)
            print(e)
            break
        temp_solve = solution[:]
        solution = [temp_solve[i] - (minus[i]) for i in range(len(temp_solve))]
        err_arr.append(f)

    if debug:
        xs = list(range(len(err_arr)))
        plt.plot(xs, err_arr)
        plt.show()

    return function(*solution), solution

def count_derive_sqr_sum(function, point : tuple):
    gradient = count_gradient(function, point, 0.00000001)
    return sum([x ** 2 for x in gradient])

class Derived_Newton:
    answer : tuple
    def __init__(self, function):
        self.function = function

    def count_error(self, point : tuple) -> float:
        self.answer = point
        # print(point, self.function(*point))
        return count_derive_sqr_sum(self.function, point)


def newton_optimize(function, point : tuple, iterations : int):
    dn = Derived_Newton(function)

    def newton_util(*point_coords) -> float:
        # print(dn.count_error(tuple(point_coords)))
        return dn.count_error(tuple(point_coords))
    print(count_gradient(newton_util, (1, 1), 0.000001))

    xs = np.arange(2, 4, 0.01)
    ys = [newton_util(__i, -7) for __i in xs]
    plt.plot(xs, ys)
    plt.show()

    __ans = find_newton_root(newton_util, point, iterations, debug=True)
    # __ans = gradient_optimize(newton_util, point, 0.000001, 0.1, iterations, debug=True)
    print(__ans)
    answer = dn.answer
    return answer



def parabola(x):
    return x ** 2

def parabaloid(x, y):
    return x ** 2 + y ** 2

def opt_test_func(x, y):
    return (x - 3) ** 4 + (y + 7) ** 2  + 10

if __name__ == '__main__':
    # print(gradient_optimize(parabaloid, (100, 100), 0.000000001, learning_rate=0.001, iterations=1000000))
    """
    newton_reses = []
    grad_reses = []
    iters = 1000
    for iter_num in range(iters):
        newton_reses.append(find_newton_root(parabaloid, (100, 100), iterations=iter_num)[0])
        grad_reses.append(gradient_optimize(parabaloid, (100, 100), 0.000000001, learning_rate=0.01, iterations=iter_num)[0])

    plt.plot(list(range(iters)), newton_reses)
    plt.plot(list(range(iters)), grad_reses)
    plt.show()
    """

    print(newton_optimize(opt_test_func, (1, 1), 10000))