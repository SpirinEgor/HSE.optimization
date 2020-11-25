import numpy
from scipy.optimize.linesearch import line_search

from assignment_2.config import Config
from assignment_2.line_searches import AbstractLineSearch, ArmijoLineSearch
from assignment_2.oracles import AbstractOracle


class WolfeLineSearch(AbstractLineSearch):
    name: str = "wolfe"

    def __init__(self, config: Config):
        super().__init__(config)
        self._armijo_line_search = ArmijoLineSearch(self._config)

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        alpha = line_search(
            oracle.value, oracle.grad, cur_point, direction, c1=self._config.armijo_c, c2=self._config.wolfe_second_c
        )[0]
        if alpha is None:
            alpha = self._armijo_line_search(oracle, cur_point, direction)
        return alpha
