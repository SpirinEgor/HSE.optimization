from typing import Optional

import numpy

from assignment_2_3_4.optimizers import AbstractOptimizer, OptimizationStep
from assignment_2_3_4.oracles import AbstractOracle


class GradientDescentOptimizer(AbstractOptimizer):
    name = "gradient descent"

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        return -last_point.grad

    def _aggregate_optimization_step(
        self,
        oracle: AbstractOracle,
        new_point: numpy.ndarray,
        passed_time: float,
        start_grad_norm: Optional[numpy.ndarray] = None,
    ) -> OptimizationStep:
        new_value, new_grad = oracle.fuse_value_grad(new_point)
        stop_criterion = 1 if start_grad_norm is None else (new_grad * new_grad).sum() / start_grad_norm
        return OptimizationStep(new_point, new_value, new_grad, passed_time, oracle.n_calls, stop_criterion)
