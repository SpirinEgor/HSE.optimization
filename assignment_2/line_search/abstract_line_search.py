from abc import ABC

import numpy

from assignment_2.oracles import AbstractOracle


class AbstractLineSearch(ABC):
    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> numpy.float:
        pass
