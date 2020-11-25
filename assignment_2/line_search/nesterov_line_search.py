from typing import Callable

import numpy

from assignment_2.line_search import AbstractLineSearch
from assignment_2.oracles import AbstractOracle


class NesterovLineSearch(AbstractLineSearch):
    def __init__(self, c: float, max_iter: int, start_point: int):
        super().__init__()
        self._c = c
        self._max_iter = max_iter
        self._start_point = start_point

    def get_alpha(self, function: Callable, direction: numpy.ndarray) -> numpy.float:
        alpha = self._start_point
        zero_value = function(0)
        direction = (direction * direction).sum()
        for _ in range(self._max_iter):
            if function(alpha) <= zero_value - self._c * alpha * direction:
                break
            alpha /= 2
        return alpha

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        minimization_function = self._get_minimization_function(oracle, cur_point, direction)
        return self.get_alpha(minimization_function, direction)
