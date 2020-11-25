from typing import Optional

import numpy
from scipy.linalg import cho_factor, LinAlgError, cho_solve

from assignment_2.config import Config
from assignment_2.optimizers import AbstractOptimizer, OptimizationStep
from assignment_2.oracles import AbstractOracle


class NewtonCholeskyOptimizer(AbstractOptimizer):
    name: str = "newton-cholesky"
    _default_tau = 1e-10

    def __init__(self, config: Config):
        super().__init__(config)
        self._tau = self._default_tau

    def reset_state(self):
        self._tau = self._default_tau

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        is_tau_changed = False
        while True:
            try:
                _cho_factor = cho_factor(last_point.hessian)
                break
            except LinAlgError:
                diag_range = numpy.arange(last_point.hessian.shape[0])
                last_point.hessian[diag_range, diag_range] += self._tau
                self._tau *= 2
                is_tau_changed = True
        if is_tau_changed:
            self._tau /= 2
        return cho_solve(_cho_factor, -last_point.grad)

    def _aggregate_optimization_step(
        self,
        oracle: AbstractOracle,
        new_point: numpy.ndarray,
        passed_time: float,
        start_grad_norm: Optional[numpy.ndarray] = None,
    ) -> OptimizationStep:
        new_value, new_grad, new_hessian = oracle.fuse_value_grad_hessian(new_point)
        stop_criterion = 1 if start_grad_norm is None else (new_grad * new_grad).sum() / start_grad_norm
        return OptimizationStep(
            new_point, new_value, new_grad, passed_time, oracle.n_calls, stop_criterion, new_hessian
        )
