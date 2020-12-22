import numpy

from assignment_2_3_4.config import Config
from assignment_2_3_4.line_searches import make_line_search, WolfeLineSearch
from assignment_2_3_4.optimizers import (
    GradientDescentOptimizer,
    HessianFreeNewtonOptimizer,
    LBFGSOptimizer,
    LassoOptimizer,
)
from assignment_2_3_4.oracles import AbstractOracle


def optimize(
    oracle: AbstractOracle,
    start_point: numpy.ndarray,
    line_search_method: str,
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> numpy.ndarray:
    config = Config()
    config.tol = tol
    config.max_iter = max_iter
    line_search = make_line_search(line_search_method, config)
    optimizer = GradientDescentOptimizer(config)
    optimization_results = optimizer.optimize(oracle, start_point, line_search)
    return optimization_results[-1].point


def hfn_optimize(
    oracle: AbstractOracle,
    start_point: numpy.ndarray,
    line_search_method: str,
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> numpy.ndarray:
    config = Config()
    config.tol = tol
    config.max_iter = max_iter
    line_search = make_line_search(line_search_method, config)
    optimizer = HessianFreeNewtonOptimizer(config)
    optimization_results = optimizer.optimize(oracle, start_point, line_search)
    return optimization_results[-1].point


def lbfgs_optimize(
    oracle: AbstractOracle, start_point: numpy.ndarray, tol: float, history_size: int = 10
) -> numpy.ndarray:
    config = Config()
    config.tol = tol
    config.l_bfgs_history_size = history_size
    line_search = WolfeLineSearch(config)
    optimizer = LBFGSOptimizer(config)
    optimization_results = optimizer.optimize(oracle, start_point, line_search)
    return optimization_results[-1].point


def optimize_lasso(
    oracle: AbstractOracle, start_point: numpy.ndarray, l1_lambda: float = 1e-2, tol: float = 1e-8
) -> numpy.ndarray:
    config = Config()
    config.tol = tol
    config.lasso_lambda = l1_lambda
    optimizer = LassoOptimizer(config)
    optimization_results = optimizer.optimize(oracle, start_point)
    return optimization_results[-1].point
