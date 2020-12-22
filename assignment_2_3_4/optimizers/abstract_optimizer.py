from abc import ABC
from time import time
from typing import List, Optional

import numpy

from assignment_2_3_4.config import Config
from assignment_2_3_4.line_searches import AbstractLineSearch
from assignment_2_3_4.optimizers.optimization_step import OptimizationStep
from assignment_2_3_4.oracles import AbstractOracle


class AbstractOptimizer(ABC):
    name: str = None

    def __init__(self, config: Config):
        self._config = config

    def reset_state(self):
        pass

    def optimize(
        self, oracle: AbstractOracle, start_point: numpy.ndarray, line_search: AbstractLineSearch = None
    ) -> List[OptimizationStep]:
        oracle.reset_call_counter()
        self.reset_state()
        if line_search is not None:
            line_search.reset_state()
        return self._optimize_oracle(oracle, start_point, line_search)

    def _optimize_oracle(
        self, oracle: AbstractOracle, start_point: numpy.ndarray, line_search: AbstractLineSearch = None
    ) -> List[OptimizationStep]:
        if line_search is None:
            raise ValueError("LineSearch is required for default optimizer behaviour")

        points = [self._aggregate_optimization_step(oracle, start_point, 0)]
        start_grad_norm = (points[-1].grad * points[-1].grad).sum()

        start_time = time()
        for n_iter in range(self._config.max_iter):
            # Check stop criterion
            if points[-1].stop_criterion <= self._config.tol:
                break
            # Calculate step size
            direction = self._get_direction(oracle, points[-1])
            step_size = line_search(oracle, points[-1].point, direction)
            # Calculate next point
            next_point = points[-1].point + step_size * direction
            points.append(self._aggregate_optimization_step(oracle, next_point, time() - start_time))
            points[-1].stop_criterion = (points[-1].grad * points[-1].grad).sum() / start_grad_norm
        return points

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        raise NotImplementedError()

    def _aggregate_optimization_step(
        self,
        oracle: AbstractOracle,
        new_point: numpy.ndarray,
        passed_time: float,
    ) -> OptimizationStep:
        new_value, new_grad = oracle.fuse_value_grad(new_point)
        return OptimizationStep(new_point, new_value, new_grad, passed_time, oracle.n_calls)
