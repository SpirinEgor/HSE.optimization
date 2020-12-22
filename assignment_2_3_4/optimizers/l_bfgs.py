from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque

import numpy

from assignment_2_3_4.config import Config
from assignment_2_3_4.optimizers import AbstractOptimizer, OptimizationStep
from assignment_2_3_4.oracles import AbstractOracle


class LBFGSOptimizer(AbstractOptimizer):
    name: str = "l-bfgs"

    @dataclass
    class _LBFGSHistory:
        s: numpy.ndarray
        y: numpy.ndarray
        st_y: float = None

        def __post_init__(self):
            self.st_y = (self.s * self.y).sum()

    _history: Deque["_LBFGSHistory"] = None
    _last_point: Optional[OptimizationStep] = None

    def __init__(self, config: Config):
        super().__init__(config)
        self._history_size = config.l_bfgs_history_size
        self._history = deque()

    def reset_state(self):
        self._history = deque()
        self._last_point = None

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        direction = -last_point.grad
        self._last_point = last_point
        if len(self._history) == 0:
            return direction
        gammas = []
        for hs in reversed(self._history):
            gammas.append((hs.s * direction).sum() / hs.st_y)
            direction -= gammas[-1] * hs.y
        direction *= self._history[-1].st_y / (self._history[-1].y * self._history[-1].y).sum()
        for hs, gamma in zip(self._history, reversed(gammas)):
            beta = (hs.y * direction).sum() / hs.st_y
            direction += (gamma - beta) * hs.s
        return direction

    def _aggregate_optimization_step(
        self,
        oracle: AbstractOracle,
        new_point: numpy.ndarray,
        passed_time: float,
        start_grad_norm: Optional[numpy.ndarray] = None,
    ) -> OptimizationStep:
        new_value, new_grad = oracle.fuse_value_grad(new_point)
        stop_criterion = 1

        if start_grad_norm is not None:
            assert self._last_point is not None, "call _get_direction before aggregating optimization step"
            self._history.append(
                self._LBFGSHistory(s=new_point - self._last_point.point, y=new_grad - self._last_point.grad)
            )
            stop_criterion = (new_grad * new_grad).sum() / start_grad_norm

        if len(self._history) > self._history_size:
            self._history.popleft()
        return OptimizationStep(new_point, new_value, new_grad, passed_time, oracle.n_calls, stop_criterion)
