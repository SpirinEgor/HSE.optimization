import numpy

from assignment_2.config import Config
from assignment_2.line_searches import make_line_search
from assignment_2.optimizers import make_optimizer
from assignment_2.oracles import AbstractOracle


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
    optimizer = make_optimizer("gradient_descent", config)
    optimization_results = optimizer.optimize(oracle, line_search, start_point)
    return optimization_results[-1].point
