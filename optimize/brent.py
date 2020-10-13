from typing import Callable, Tuple

import numpy

from optimize import OptimizeResult
from utils import linear_approximation


class Brent:

    _golden_value = 0.381966

    def __init__(self, max_iterations: int = 1e5):
        self._max_iterations = max_iterations

    def brent_with_derivatives(
            self, oracle: Callable[[float], Tuple[float, float]], a: float, c: float, eps: float
    ) -> OptimizeResult:
        """http://www.machinelearning.ru/wiki/images/a/a8/MOMO12_min1d.pdf
        """
        _n_iterations = 0
        _history = []

        # Init block
        first_min = second_min = prev_sec_min = (a + c) / 2
        f_first_min, df_first_min = oracle(first_min)
        f_second_min, df_second_min = f_first_min, df_first_min
        f_prev_sec_min, df_prev_sec_min = f_first_min, df_first_min
        current_step = previous_step = c - a

        while _n_iterations < self._max_iterations:
            _n_iterations += 1
            _history.append(first_min)

            # handle convergence:
            if numpy.abs(df_first_min) <= eps or current_step <= eps:
                return OptimizeResult(first_min, _history, _n_iterations)

            temp_step = previous_step
            previous_step = current_step
            current_step = None
            next_min = None

            # first parabola
            if first_min != second_min and df_first_min != df_second_min:
                possible_min = linear_approximation(first_min, df_first_min, second_min, df_second_min)
                if a + eps <= possible_min <= c - eps and numpy.abs(possible_min - first_min) < temp_step / 2:
                    next_min = possible_min
                    current_step = numpy.abs(next_min - first_min)

            # second parabola
            if first_min != prev_sec_min and df_first_min != df_prev_sec_min:
                possible_min = linear_approximation(first_min, df_first_min, prev_sec_min, df_prev_sec_min)
                if a + eps <= possible_min <= c - eps and numpy.abs(possible_min - first_min) < previous_step / 2:
                    if next_min is None or numpy.abs(possible_min - first_min) < current_step:
                        next_min = possible_min
                        current_step = numpy.abs(next_min - first_min)

            # bisect
            if next_min is None:
                next_min = (a + first_min) / 2 if df_first_min > 0 else (first_min + c) / 2
                current_step = numpy.abs(next_min - first_min)

            # check min step size
            if current_step < eps:
                next_min = first_min + numpy.sign(next_min - first_min) * eps
                current_step = eps

            # ask oracle for new values
            f_next_min, df_next_min = oracle(next_min)

            # update brackets and points
            if f_next_min <= f_first_min:
                a, c = (first_min, c) if next_min >= first_min else (a, first_min)

                prev_sec_min, f_prev_sec_min, df_prev_sec_min = second_min, f_second_min, df_second_min
                second_min, f_second_min, df_second_min = first_min, f_first_min, df_first_min
                first_min, f_first_min, df_first_min = next_min, f_next_min, df_next_min
            else:
                a, c = (a, next_min) if next_min >= first_min else (next_min, c)
                if f_next_min <= f_second_min or second_min == first_min:
                    prev_sec_min, f_prev_sec_min, df_prev_sec_min = second_min, f_second_min, df_second_min
                    second_min, f_second_min, df_second_min = next_min, f_next_min, df_next_min
                elif f_next_min <= f_prev_sec_min or prev_sec_min == first_min or prev_sec_min == second_min:
                    prev_sec_min, f_prev_sec_min, df_prev_sec_min = next_min, f_next_min, df_next_min

        return OptimizeResult(first_min, _history, _n_iterations)

    def get_optimize_function(self):
        return self.brent_with_derivatives
