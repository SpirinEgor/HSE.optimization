from typing import Callable

import numpy
from scipy.constants import golden

from assignment_2.line_search import AbstractLineSearch
from assignment_2.oracles import AbstractOracle


class GoldenLineSearch(AbstractLineSearch):
    def __init__(self, a: numpy.float, b: numpy.float, tol: float, max_iter: int):
        self._a = min(a, b)
        self._b = max(a, b)
        self._tol = tol
        self._max_iter = max_iter

    def golden(self, function: Callable) -> numpy.float:
        left, right = self._a, self._b
        step = 2 - golden
        x1 = self._a + step * (right - left)
        x2 = self._b - step * (right - left)
        f_x1 = function(x1)
        f_x2 = function(x2)
        for _ in range(self._max_iter):
            if x2 - x1 < self._tol:
                break
            if f_x1 < f_x2:
                right = x2
                x2, f_x2 = x1, f_x1
                x1 = left + (right - left) * step
                f_x1 = function(x1)
            else:
                left = x1
                x1, f_x1 = x2, f_x2
                x2 = right - (right - left) * step
                f_x2 = function(x2)
        return (x1 + x2) / 2

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        minimize_function = self._get_minimization_function(oracle, cur_point, direction)
        return self.golden(minimize_function)
