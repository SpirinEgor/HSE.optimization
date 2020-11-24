import numpy

from assignment_2.line_search import AbstractLineSearch, GoldenLineSearch
from assignment_2.oracles import AbstractOracle


def get_line_search_method(name: str) -> AbstractLineSearch:
    if name == "golden":
        return GoldenLineSearch(-15, 15)
    else:
        raise ValueError(f"Unknown line search algorithm '{name}'")


def square_norm(matrix: numpy.ndarray) -> numpy.ndarray:
    return numpy.square(matrix).sum()


def optimize(
        oracle: AbstractOracle,
        start_point: numpy.ndarray,
        line_search_method: str,
        tol: float = 1e-8,
        max_iter: int = 10000
) -> numpy.ndarray:
    line_search_method = get_line_search_method(line_search_method)
    points = [start_point]
    zero_grad = oracle.grad(start_point)
    zero_grad_square_norm = square_norm(zero_grad)
    for n_iter in range(max_iter):
        cur_point_value, cur_point_grad = oracle.fuse_value_grad(points[-1])
        cur_grad_square_norm = square_norm(cur_point_grad)
        if cur_grad_square_norm <= tol * zero_grad_square_norm:
            break
        step_size = line_search_method(oracle, points[-1], -cur_point_grad)
        points.append(points[-1] - step_size * cur_point_grad)
    return points[-1]
