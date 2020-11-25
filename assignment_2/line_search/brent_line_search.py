import numpy
from scipy.optimize import brent

from assignment_2.line_search import AbstractLineSearch
from assignment_2.oracles import AbstractOracle


class BrentLineSearch(AbstractLineSearch):
    def __init__(self, a: float, c: float, tol: float, max_iter: int):
        super().__init__()
        self._bracket = (a, c)
        self._tol = tol
        self._max_iter = max_iter

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        minimization_function = self._get_minimization_function(oracle, cur_point, direction)
        return brent(
            minimization_function,
            brack=self._bracket,
            tol=self._tol,
            maxiter=self._max_iter,
        )
