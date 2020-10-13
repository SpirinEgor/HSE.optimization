from dataclasses import dataclass
from typing import List


@dataclass
class OptimizeResult:
    x_min: float
    optimize_history: List[float]
    n_iterations: int
