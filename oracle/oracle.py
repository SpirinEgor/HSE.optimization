from abc import ABC
from typing import Callable, Tuple


class Oracle(ABC):
    _name = None
    _x_min = None

    def get_oracle(self) -> Callable[[float], Tuple[float, float]]:
        raise NotImplementedError

    def get_name(self) -> str:
        return self._name

    def get_x_min(self) -> float:
        return self._x_min
