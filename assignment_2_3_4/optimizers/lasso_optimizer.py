from time import time
from typing import List, Optional

import numpy

from assignment_2_3_4.line_searches import AbstractLineSearch
from assignment_2_3_4.optimizers import AbstractOptimizer, OptimizationStep
from assignment_2_3_4.oracles import AbstractOracle


class LassoOptimizer(AbstractOptimizer):
    name: str = "lasso"

    def optimize(
        self, oracle: AbstractOracle, line_search: AbstractLineSearch, start_point: numpy.ndarray
    ) -> List[OptimizationStep]:
        oracle.reset_call_counter()
        line_search.reset_state()
        self.reset_state()

        points = [self._aggregate_optimization_step(oracle, start_point, 0)]

        cur_l = self._config.lasso_start_l
        start_time = time()
        for _ in range(self._config.max_iter):
            next_point = self._lasso_proximal(points[-1].point - points[-1].grad / cur_l, 1.0 / cur_l)

            for _ in range(self._config.max_iter):
                next_value = oracle.value(next_point)
                diff = next_point - points[-1].point
                if next_value < points[-1].value + points[-1].grad.dot(diff) + cur_l / 2 * numpy.linalg.norm(diff):
                    break
                cur_l *= 2

            cur_l = max(self._config.lasso_start_l, cur_l / 2)
            points.append(self._aggregate_optimization_step(oracle, next_point, time() - start_time))
            if cur_l * numpy.linalg.norm(points[-2].value - points[-1].value) < self._config.tol:
                break

        return points

    def _lasso_proximal(self, point: numpy.ndarray, alpha: float) -> numpy.ndarray:
        return numpy.sign(point) * numpy.maximum(point - alpha * self._config.lasso_lambda, 0)

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

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        pass
