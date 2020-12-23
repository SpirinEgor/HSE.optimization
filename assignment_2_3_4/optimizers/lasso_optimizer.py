from time import time
from typing import List

import numpy

from assignment_2_3_4.line_searches import AbstractLineSearch
from assignment_2_3_4.optimizers import AbstractOptimizer, OptimizationStep
from assignment_2_3_4.oracles import AbstractOracle


class LassoOptimizer(AbstractOptimizer):
    name: str = "lasso"

    def _optimize_oracle(
        self, oracle: AbstractOracle, start_point: numpy.ndarray, line_search: AbstractLineSearch = None
    ) -> List[OptimizationStep]:
        points = [self._aggregate_optimization_step(oracle, start_point, 0)]

        lipschitz: float = 1.0
        start_time = time()
        for _ in range(self._config.max_iter):
            for _ in range(self._config.max_iter):
                alpha = 1 / lipschitz
                prox = self._lasso_proximal(points[-1].point - alpha * points[-1].grad, alpha)
                delta = prox - points[-1].point
                delta_norm = delta @ delta
                if oracle.value(prox) <= points[-1].value + points[-1].grad @ delta + lipschitz / 2 * delta_norm:
                    break
                lipschitz *= 2

            points.append(self._aggregate_optimization_step(oracle, prox, time() - start_time))
            points[-1].stop_criterion = delta_norm / (alpha ** 2)
            if points[-1].stop_criterion <= self._config.tol:
                break
            lipschitz /= 2

        return points

    def _lasso_proximal(self, point: numpy.ndarray, alpha: float) -> numpy.ndarray:
        return numpy.sign(point) * numpy.maximum(numpy.abs(point) - alpha * self._config.lasso_lambda, 0)

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        pass
