from typing import Callable, Tuple


class Brent:
    def __init__(self, max_iterations: int = 1e5):
        self._max_iterations = max_iterations
        self._history = []

    def brent_with_derivatives(
            self, oracle: Callable[[float], Tuple[float, float]], a: float, b: float, eps: float
    ) -> float:
        return 0

    def get_optimize_function(self):
        return self.brent_with_derivatives
