from dataclasses import dataclass
from typing import Optional

import numpy


@dataclass
class OptimizationStep:
    point: numpy.ndarray
    value: numpy.float
    grad: numpy.ndarray

    passed_time: float
    oracle_calls: int
    stop_criterion: Optional[float] = 1.0

    hessian: Optional[numpy.ndarray] = None
