import numpy
from scipy.optimize.linesearch import line_search

from assignment_2.line_searches import AbstractLineSearch, ArmijoLineSearch
from assignment_2.oracles import AbstractOracle


class WolfeLineSearch(AbstractLineSearch):
    def __init__(self, c1: float, c2: float, max_iter: int, armijo_start_point: float):
        super().__init__()
        self._c1 = c1
        self._c2 = c2
        self._max_iter = max_iter
        self._armijo_start_point = armijo_start_point

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        alpha = line_search(oracle.value, oracle.grad, cur_point, direction, c1=self._c1, c2=self._c2)[0]
        if alpha is None:
            armijo_line_search = ArmijoLineSearch(self._c1, self._max_iter, self._armijo_start_point)
            alpha = armijo_line_search(oracle, cur_point, direction)
        return alpha
