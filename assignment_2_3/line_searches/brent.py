import numpy
from scipy.optimize import brent

from assignment_2_3.line_searches import AbstractLineSearch
from assignment_2_3.oracles import AbstractOracle


class BrentLineSearch(AbstractLineSearch):
    name: str = "brent"

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> float:
        minimization_function = self._get_minimization_function(oracle, cur_point, direction)
        return brent(
            minimization_function,
            brack=(self._config.bracket_left, self._config.bracket_right),
            tol=self._config.tol,
            maxiter=self._config.max_iter_line_search,
        )
