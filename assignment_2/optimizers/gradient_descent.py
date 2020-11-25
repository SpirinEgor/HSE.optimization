import numpy

from assignment_2.optimizers.abstract_optimizer import AbstractOptimizer
from assignment_2.optimizers.optimization_step import OptimizationStep
from assignment_2.oracles import AbstractOracle


class GradientDescentOptimizer(AbstractOptimizer):
    name = "gradient descent"

    def _get_direction(self, last_point: OptimizationStep) -> numpy.ndarray:
        return -last_point.grad

    def _aggregate_optimization_step(
        self, oracle: AbstractOracle, new_point: numpy.ndarray, passed_time: float, start_grad_norm: numpy.ndarray
    ) -> OptimizationStep:
        new_value, new_grad = oracle.fuse_value_grad(new_point)
        stop_criterion = (new_grad * new_grad).sum() / start_grad_norm
        return OptimizationStep(new_point, new_value, new_grad, passed_time, oracle.n_calls, stop_criterion)
