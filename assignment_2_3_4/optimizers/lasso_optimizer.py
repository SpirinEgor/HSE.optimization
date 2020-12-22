from time import time
from typing import List, Optional

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

        cur_l = self._config.lasso_start_l
        start_time = time()
        for _ in range(self._config.max_iter):
            if points[-1].stop_criterion < self._config.tol:
                break

            next_point = self._lasso_proximal(points[-1].point - points[-1].grad / cur_l, 1.0 / cur_l)

            for _ in range(self._config.max_iter):
                next_value = oracle.value(next_point)
                diff = next_point - points[-1].point
                diff_norm = numpy.linalg.norm(diff)
                if next_value < points[-1].value + points[-1].grad.dot(diff) + cur_l / 2 * diff_norm * diff_norm:
                    break
                cur_l *= 2

            points.append(self._aggregate_optimization_step(oracle, next_point, time() - start_time))
            points[-1].stop_criterion = cur_l * numpy.linalg.norm(points[-2].value - points[-1].value)
            points[-1].stop_criterion *= points[-1].stop_criterion
            cur_l = max(self._config.lasso_start_l, cur_l / 2)

        return points

    def _lasso_proximal(self, point: numpy.ndarray, alpha: float) -> numpy.ndarray:
        return numpy.sign(point) * numpy.maximum(point - alpha * self._config.lasso_lambda, 0)

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        pass
