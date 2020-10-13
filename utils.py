from typing import Callable, Tuple


class CountCallsWrapper:
    def __init__(self, oracle: Callable[[float], Tuple[float, float]]):
        self.oracle = oracle
        self.call_counter = 0

    def __call__(self, x: float) -> Tuple[float, float]:
        self.call_counter += 1
        return self.oracle(x)

    def get_number_of_calls(self):
        return self.call_counter


def linear_approximation(x1: float, y1: float, x2: float, y2: float) -> float:
    k = (y2 - y1) / (x2 - x1)
    b = y2 - k * x2
    x3 = -b / k
    return x3


def get_derivative_in_point(f: Callable[[float], float], x: float, delta: float = 1e-8) -> float:
    return (f(x + delta) - f(x)) / delta
