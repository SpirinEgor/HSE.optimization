from typing import Callable, Tuple


class OracleWrapper:
    def __init__(self, oracle: Callable[[float], Tuple[float, float]]):
        self.oracle = oracle
        self.call_counter = 0

    def __call__(self, x: float) -> Tuple[float, float]:
        self.call_counter += 1
        return self.oracle(x)

    def get_number_of_calls(self):
        return self.call_counter
