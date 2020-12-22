from typing import Callable

import numpy

from assignment_2_3_4.line_searches import AbstractLineSearch
from assignment_2_3_4.oracles import AbstractOracle


class ArmijoLineSearch(AbstractLineSearch):
    name: str = "armijo"

    def _backtracking(self, function: Callable, grad_function: Callable, direction: numpy.ndarray) -> float:
        alpha = self._config.bracket_right
        zero_value = function(0)
        zero_grad = (grad_function(0) * direction).sum()
        for _ in range(self._config.max_iter_line_search):
            if function(alpha) <= zero_value + self._config.armijo_c * alpha * zero_grad:
                break
            alpha /= 2
        return alpha

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> float:
        minimization_function = self._get_minimization_function(oracle, cur_point, direction)
        grad_function = self._get_grad_function(oracle, cur_point, direction)
        return self._backtracking(minimization_function, grad_function, direction)
