from typing import Tuple

import numpy
from scipy.optimize import golden

from assignment_2.line_search import AbstractLineSearch
from assignment_2.oracles import AbstractOracle


class GoldenLineSearch(AbstractLineSearch):
    def __init__(self, a: numpy.float, c: numpy.float):
        self._brackets = (a, c)

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        def minimization_function(alpha: numpy.float) -> numpy.float:
            return oracle.value(cur_point + alpha * direction)
        return golden(minimization_function, brack=self._brackets)
