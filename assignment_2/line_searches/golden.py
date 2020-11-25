from typing import Callable

import numpy
from scipy.constants import golden

from assignment_2.line_searches import AbstractLineSearch
from assignment_2.oracles import AbstractOracle


class GoldenLineSearch(AbstractLineSearch):
    name: str = "golden"

    def golden(self, function: Callable) -> numpy.float:
        left, right = self._config.bracket_left, self._config.bracket_right
        step = 2 - golden
        x1 = left + step * (right - left)
        x2 = right - step * (right - left)
        f_x1 = function(x1)
        f_x2 = function(x2)
        for _ in range(self._config.max_iter_line_search):
            if x2 - x1 < self._config.tol:
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
