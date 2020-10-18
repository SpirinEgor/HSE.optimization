from dataclasses import dataclass

import numpy

from optimize import IBrent, BrentMomo, BrentNumericalRecipes
from oracle import Oracle, unimodal
from utils import CountCallsWrapper

LINE_DELIMITER = "=" * 100


@dataclass
class OptimizeParameters:
    left_bound: float
    right_bound: float
    epsilon: float = 1e-8


def test_optimize(brent: IBrent, params: OptimizeParameters, oracle: Oracle):
    oracle_function = oracle.get_oracle()
    count_call_wrapper = CountCallsWrapper(oracle_function)
    optimize_result = brent.brent_with_derivatives(
        count_call_wrapper, params.left_bound, params.right_bound, params.epsilon
    )
    x_min = optimize_result.x_min
    x_min_true = oracle.get_x_min()
    error = numpy.abs(x_min - oracle.get_x_min())
    error = 0 if error < params.epsilon else error
    print(LINE_DELIMITER)
    print(f"Optimized function: {oracle.get_name()}\n"
          f"Reached minimum: x={x_min} (f(x), f'(x) = {oracle_function(x_min)})\n"
          f"   True minimum: x={x_min_true} (f(x), f'(x) = {oracle_function(x_min_true)})\n"
          f"Error: {error}\n"
          f"Number of calls to the oracle: {count_call_wrapper.get_number_of_calls()}\n"
          f"Optimization history: {' '.join(map(str, optimize_result.optimize_history))}")
    print(LINE_DELIMITER)


def main():
    test_samples = [
        (OptimizeParameters(-10, 10), unimodal.Example()),
        (OptimizeParameters(-10, 10), unimodal.Quadratic(5)),
        (OptimizeParameters(0, 3), unimodal.Function4()),
        (OptimizeParameters(-1, 2), unimodal.Function13()),
        (OptimizeParameters(0, 6), unimodal.Function18())
    ]

    max_iterations = 20
    # brent = BrentMomo(max_iterations)
    brent = BrentNumericalRecipes(max_iterations)

    for optimizer_params, oracle in test_samples:
        test_optimize(brent, optimizer_params, oracle)


if __name__ == "__main__":
    main()
