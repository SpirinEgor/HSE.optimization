from typing import Callable

import numpy

from assignment_2.line_searches import AbstractLineSearch
from assignment_2.oracles import AbstractOracle


class ArmijoLineSearch(AbstractLineSearch):
    def __init__(self, c: float, max_iter: int, start_point: float):
        super().__init__()
        self._c = c
        self._max_iter = max_iter
        self._start_point = start_point

    def _backtracking(self, function: Callable, grad_function: Callable, direction: numpy.ndarray) -> numpy.float:
        alpha = self._start_point
        zero_value = function(0)
        zero_grad = (grad_function(0) * direction).sum()
        for _ in range(self._max_iter):
            if function(alpha) <= zero_value + self._c * alpha * zero_grad:
                break
            alpha /= 2
        return alpha

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        minimization_function = self._get_minimization_function(oracle, cur_point, direction)
        grad_function = self._get_grad_function(oracle, cur_point, direction)
        return self._backtracking(minimization_function, grad_function, direction)
