from typing import Callable, Tuple

import numpy

from oracle import Oracle
from utils import get_derivative_in_point


__all__ = ["Function2", "Function5", "Function6", "Function10", "Function15"]


class Function2(Oracle):
    """Problem02: http://infinity77.net/global_optimization/test_functions_1d.html
    """
    _name = "sin(x) + sin(x * 10 / 3)"
    _x_min = 5.145735
    _f_x_min = -1.899599
    _eps = 1e-6

    @staticmethod
    def _f(x: float) -> float:
        first_term = numpy.sin(x)
        second_term = numpy.sin(x * 10.0 / 3.0)
        return first_term + second_term

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return self._f(x), get_derivative_in_point(self._f, x)
        return f


class Function5(Oracle):
    """Problem05: http://infinity77.net/global_optimization/test_functions_1d.html
    """
    _name = "-(1.4 - 3x) * sin(18x)"
    _x_min = 0.96609
    _f_x_min = -1.48907
    _eps = 1e-5

    @staticmethod
    def _f(x: float) -> float:
        first_term = 1.4 - 3 * x
        second_term = numpy.sin(18 * x)
        return -first_term * second_term

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return self._f(x), get_derivative_in_point(self._f, x)
        return f


class Function6(Oracle):
    """Problem06: http://infinity77.net/global_optimization/test_functions_1d.html
    """
    _name = "-[x + sin(x)] * e^{-x^2}"
    _x_min = 0.67956
    _f_x_min = -0.824239
    _eps = 1e-5

    @staticmethod
    def _f(x: float) -> float:
        first_term = x + numpy.sin(x)
        second_term = numpy.exp(-x * x)
        return -first_term * second_term

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return self._f(x), get_derivative_in_point(self._f, x)
        return f


class Function10(Oracle):
    """Problem10: http://infinity77.net/global_optimization/test_functions_1d.html
    """
    _name = "-x * sin(x)"
    _x_min = 7.9787
    _f_x_min = -7.916727
    _eps = 1e-4

    @staticmethod
    def _f(x: float) -> float:
        return -x * numpy.sin(x)

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return self._f(x), get_derivative_in_point(self._f, x)
        return f


class Function15(Oracle):
    """Problem15: http://infinity77.net/global_optimization/test_functions_1d.html
    """
    _name = "(x^2 - 5x + 6) / (x^2 + 1)"
    _x_min = 2.41422
    _f_x_min = -0.03553
    _eps = 1e-5

    @staticmethod
    def _f(x: float) -> float:
        return (x * x - 5 * x + 6) / (x * x + 1)

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return self._f(x), get_derivative_in_point(self._f, x)
        return f
