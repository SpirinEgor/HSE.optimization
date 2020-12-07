import numpy

from assignment_2_3.config import Config
from assignment_2_3.line_searches import make_line_search, WolfeLineSearch
from assignment_2_3.optimizers import GradientDescentOptimizer, HessianFreeNewtonOptimizer, LBFGSOptimizer
from assignment_2_3.oracles import AbstractOracle


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
    optimization_results = optimizer.optimize(oracle, line_search, start_point)
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
    optimization_results = optimizer.optimize(oracle, line_search, start_point)
    return optimization_results[-1].point


def lbfgs_optimize(
    oracle: AbstractOracle, start_point: numpy.ndarray, tol: float, history_size: int = 10
) -> numpy.ndarray:
    config = Config()
    config.tol = tol
    config.l_bfgs_history_size = history_size
    line_search = WolfeLineSearch(config)
    optimizer = LBFGSOptimizer(config)
    optimization_results = optimizer.optimize(oracle, line_search, start_point)
    return optimization_results[-1].point
