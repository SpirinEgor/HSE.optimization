from abc import ABC
from typing import Callable

import numpy

from assignment_2_3_4.config import Config
from assignment_2_3_4.oracles import AbstractOracle


class AbstractLineSearch(ABC):
    name: str = None

    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def _get_minimization_function(
        oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray
    ) -> Callable:
        def minimization_function(alpha: numpy.float) -> numpy.float:
            return oracle.value(cur_point + alpha * direction)

        return minimization_function

    @staticmethod
    def _get_grad_function(oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> Callable:
        def grad_function(alpha: numpy.float) -> numpy.ndarray:
            return oracle.grad(cur_point + alpha * direction)

        return grad_function

    def reset_state(self):
        pass

    def __call__(self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray) -> float:
        raise NotImplementedError()
