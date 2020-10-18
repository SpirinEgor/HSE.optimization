from abc import ABC
from typing import Callable, Tuple

from optimize import OptimizeResult


class IBrent(ABC):
    _golden_value = 0.381966

    def __init__(self, max_iterations: int = 1e5):
        self._max_iterations = max_iterations

    def get_optimize_function(self) -> Callable:
        return self.brent_with_derivatives

    @staticmethod
    def _init_brent_variables(oracle: Callable[[float], Tuple[float, float]], a: float, c: float) -> Tuple:
        first_min = second_min = prev_sec_min = (a + c) / 2
        f_first_min, df_first_min = oracle(first_min)
        f_second_min, df_second_min = f_first_min, df_first_min
        f_prev_sec_min, df_prev_sec_min = f_first_min, df_first_min
        return (
            (first_min, second_min, prev_sec_min),
            (f_first_min, f_second_min, f_prev_sec_min),
            (df_first_min, df_second_min, df_prev_sec_min)
        )

    def brent_with_derivatives(
            self, oracle: Callable[[float], Tuple[float, float]], a: float, c: float, eps: float
    ) -> OptimizeResult:
        raise NotImplementedError

