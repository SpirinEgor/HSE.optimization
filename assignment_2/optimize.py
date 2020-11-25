from dataclasses import dataclass
from typing import List
from time import time

import numpy

from assignment_2.config import Config
from assignment_2.line_searches import (
    AbstractLineSearch,
    GoldenLineSearch,
    BrentLineSearch,
    WolfeLineSearch,
    ArmijoLineSearch,
    NesterovLineSearch,
)
from assignment_2.oracles import AbstractOracle


def get_line_search_method(name: str) -> AbstractLineSearch:
    if name == "golden":
        return GoldenLineSearch(Config.bracket_left, Config.bracket_right, Config.tol, Config.iter_line_search)
    elif name == "brent":
        return BrentLineSearch(Config.bracket_left, Config.bracket_right, Config.tol, Config.iter_line_search)
    elif name == "armijo":
        return ArmijoLineSearch(Config.armijo_c, Config.iter_line_search, Config.bracket_right)
    elif name == "wolfe":
        return WolfeLineSearch(Config.armijo_c, Config.wolfe_second_c, Config.iter_line_search, Config.bracket_right)
    elif name == "nesterov":
        return NesterovLineSearch(Config.nesterov_c, Config.iter_line_search, Config.bracket_right)
    else:
        raise ValueError(f"Unknown line search algorithm '{name}'")


@dataclass
class OptimizationStep:
    point: numpy.ndarray
    value: numpy.float
    grad: numpy.ndarray
    passed_time: float
    oracle_calls: int
    stop_criterion: float


def gradient_descent_optimization(
    oracle: AbstractOracle,
    line_search: AbstractLineSearch,
    start_point: numpy.ndarray,
    tol: float,
    max_iter: int,
) -> List[OptimizationStep]:
    oracle.reset_call_counter()

    start_value, start_grad = oracle.fuse_value_grad(start_point)
    points = [OptimizationStep(start_point, start_value, start_grad, 0, oracle.n_calls, 1)]
    start_grad_norm = (start_grad * start_grad).sum()

    start_time = time()
    for n_iter in range(max_iter):
        # Check stop criterion
        if points[-1].stop_criterion <= tol:
            break
        # Calculate step size
        step_size = line_search(oracle, points[-1].point, -points[-1].grad)
        # Calculate next point
        next_point = points[-1].point - step_size * points[-1].grad
        next_value, next_grad = oracle.fuse_value_grad(next_point)
        stop_criterion = (next_grad * next_grad).sum() / start_grad_norm
        # Add to history
        points.append(
            OptimizationStep(next_point, next_value, next_grad, time() - start_time, oracle.n_calls, stop_criterion)
        )
    return points


def optimize(
    oracle: AbstractOracle,
    start_point: numpy.ndarray,
    line_search_method: str,
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> numpy.ndarray:
    line_search = get_line_search_method(line_search_method)
    optimization_results = gradient_descent_optimization(oracle, line_search, start_point, tol, max_iter)
    return optimization_results[-1].point
