import numpy

from assignment_2_3_4.optimizers import AbstractOptimizer, OptimizationStep
from assignment_2_3_4.oracles import AbstractOracle


class GradientDescentOptimizer(AbstractOptimizer):
    name = "gradient descent"

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        return -last_point.grad
