from abc import ABC
from time import time
from typing import List, Optional

import numpy

from assignment_2.config import Config
from assignment_2.line_searches import AbstractLineSearch
from assignment_2.optimizers.optimization_step import OptimizationStep
from assignment_2.oracles import AbstractOracle


class AbstractOptimizer(ABC):
    name: str = None

    def __init__(self, config: Config):
        self._config = config

    def reset_state(self):
        pass

    def optimize(
        self, oracle: AbstractOracle, line_search: AbstractLineSearch, start_point: numpy.ndarray
    ) -> List[OptimizationStep]:
        oracle.reset_call_counter()
        line_search.reset_state()
        self.reset_state()

        points = [self._aggregate_optimization_step(oracle, start_point, 0)]
        start_grad_norm = (points[-1].grad * points[-1].grad).sum()

        start_time = time()
        for n_iter in range(self._config.max_iter):
            # Check stop criterion
            if points[-1].stop_criterion <= self._config.tol:
                break
            # Calculate step size
            direction = self._get_direction(points[-1])
            step_size = line_search(oracle, points[-1].point, direction)
            # Calculate next point
            next_point = points[-1].point + step_size * direction
            points.append(self._aggregate_optimization_step(oracle, next_point, time() - start_time, start_grad_norm))
        return points

    def _get_direction(self, last_point: OptimizationStep) -> numpy.ndarray:
        raise NotImplementedError()

    def _aggregate_optimization_step(
        self,
        oracle: AbstractOracle,
        new_point: numpy.ndarray,
        passed_time: float,
        start_grad_norm: Optional[numpy.ndarray] = None,
    ) -> OptimizationStep:
        raise NotImplementedError()
