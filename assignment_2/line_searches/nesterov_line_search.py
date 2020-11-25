from typing import Callable

import numpy

from assignment_2.config import Config
from assignment_2.line_searches import AbstractLineSearch
from assignment_2.oracles import AbstractOracle


class NesterovLineSearch(AbstractLineSearch):
    name: str = "nesterov"

    def __init__(self, config: Config):
        super().__init__(config)
        self._alpha = self._config.bracket_right

    def reset_state(self):
        super().reset_state()
        self._alpha = self._config.bracket_right

    def get_alpha(self, function: Callable, direction: numpy.ndarray) -> numpy.float:
        alpha = self._alpha
        zero_value = function(0)
        direction = (direction * direction).sum()
        for _ in range(self._config.max_iter_line_search):
            if function(alpha) <= zero_value - self._config.nesterov_c * alpha * direction:
                break
            alpha /= 2
        self._alpha = alpha * 2
        return alpha

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        minimization_function = self._get_minimization_function(oracle, cur_point, direction)
        return self.get_alpha(minimization_function, direction)
