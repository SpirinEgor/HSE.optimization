from typing import Callable, Tuple

import numpy

from oracle import Oracle

from utils import get_derivative_in_point


__all__ = ["Quadratic", "Example", "Function4", "Function13", "Function18"]


class Example(Oracle):
    """Пусть дана функция (оракул) f
    Со следующим интерфейсом
    def f(x):
        return f(x), f'(x)
    Например, оракул для квадратичной фунции x^2/2
    """
    _name = "x^2"
    _x_min = 0

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return x * x / 2, x
        return f


class Quadratic(Oracle):

    _name = "(x - k)^2"

    def __init__(self, x_min: float):
        self._x_min = x_min

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return (x - self._x_min) ** 2, 2 * (x - self._x_min)
        return f


class Function4(Oracle):
    """Problem04: http://infinity77.net/global_optimization/test_functions_1d.html
    """
    _name = "-(16x^2 - 24x + 5) * e^{-x}"
    _x_min = 2.868034
    _f_x_min = -3.85045
    _eps = 1e-6

    @staticmethod
    def _f(x: float) -> float:
        polynom = 16 * x * x - 24 * x + 5
        exp = numpy.exp(-x)
        return -polynom * exp

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return self._f(x), get_derivative_in_point(self._f, x)
        return f


class Function13(Oracle):
    """Problem13: http://infinity77.net/global_optimization/test_functions_1d.html
    """
    _name = "-x^{2/3} - (1 - x^2)^{1/3}"
    _x_min = 1 / numpy.sqrt(2)
    _f_x_min = -1.5874

    @staticmethod
    def _f(x: float) -> float:
        first_term = numpy.power(x, 2 / 3)
        in_bracket = (1 - x * x)
        second_term = numpy.sign(in_bracket) * numpy.power(numpy.abs(in_bracket), 1 / 3)
        return -first_term - second_term

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return self._f(x), get_derivative_in_point(self._f, x)
        return f


class Function18(Oracle):
    """Problem18: http://infinity77.net/global_optimization/test_functions_1d.html
    """
    _name = "{ (x-2)^2, if x<=3 } " \
            "{ 2 log(x-2) + 1, otherwise }"
    _x_min = 2
    _f_x_min = 0

    @staticmethod
    def _f(x: float) -> float:
        if x <= 3:
            return (x - 2) ** 2
        return 2 * numpy.log(x - 2) + 1

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        def f(x):
            return self._f(x), get_derivative_in_point(self._f, x)
        return f
