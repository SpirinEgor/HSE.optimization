from abc import ABC
from typing import Callable

import numpy

from assignment_2.oracles import AbstractOracle


class AbstractLineSearch(ABC):
    @staticmethod
    def _get_minimization_function(
        oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray
    ) -> Callable:
        def minimization_function(alpha: numpy.float) -> numpy.float:
            return oracle.value(cur_point + alpha * direction)

        return minimization_function

    @staticmethod
    def _get_grad_function(
        oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray
    ) -> Callable:
        def grad_function(alpha: numpy.float) -> numpy.ndarray:
            return oracle.grad(cur_point + alpha * direction)

        return grad_function

    def __call__(
        self, oracle: AbstractOracle, cur_point: numpy.ndarray, direction: numpy.ndarray
    ) -> numpy.float:
        raise NotImplementedError()
