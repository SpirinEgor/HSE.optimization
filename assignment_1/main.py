import numpy

from assignment_1.optimize import IBrent, BrentNumericalRecipes
from assignment_1.oracle import Oracle, continuous
from assignment_1.oracle import unimodal
from assignment_1.utils import CountCallsWrapper, OptimizeParameters

LINE_DELIMITER = "=" * 100


def _round_by_epsilon(number: float, epsilon: float) -> float:
    """Assume epsilon=1e-x"""
    eps = int(numpy.log10(epsilon))
    return round(number, -eps)


def test_optimize(brent: IBrent, params: OptimizeParameters, oracle: Oracle):
    oracle_function = oracle.get_oracle()
    count_call_wrapper = CountCallsWrapper(oracle_function)
    optimize_result = brent.brent_with_derivatives(
        count_call_wrapper, params.left_bound, params.right_bound, oracle.get_eps()
    )
    x_min = _round_by_epsilon(optimize_result.x_min, oracle.get_eps())
    x_min_true = oracle.get_x_min()
    error = numpy.abs(x_min - oracle.get_x_min())
    error = (
        0 if error < oracle.get_eps() else _round_by_epsilon(error, oracle.get_eps())
    )
    print(LINE_DELIMITER)
    print(
        f"Optimized function: {oracle.get_name()}\n"
        f"Reached minimum: x={x_min} (f(x), f'(x) = {oracle_function(x_min)})\n"
        f"   True minimum: x={x_min_true} (f(x), f'(x) = {oracle_function(x_min_true)})\n"
        f"Error: {error}\n"
        f"Number of calls to the oracle: {count_call_wrapper.get_number_of_calls()}\n"
        f"Optimization history: {' '.join(map(str, optimize_result.optimize_history))}"
    )
    print(LINE_DELIMITER)


def main():
    test_samples = [
        (OptimizeParameters(-10, 10), unimodal.Example()),
        (OptimizeParameters(-10, 10), unimodal.Quadratic(5)),
        (OptimizeParameters(0, 10), unimodal.Function4()),
        (OptimizeParameters(-1, 2), unimodal.Function13()),
        (OptimizeParameters(0, 6), unimodal.Function18()),
        (OptimizeParameters(2.7, 7.5), continuous.Function2()),
        (OptimizeParameters(0, 2), continuous.Function5()),
        (OptimizeParameters(-10, 10), continuous.Function6()),
        (OptimizeParameters(0, 10), continuous.Function10()),
        (OptimizeParameters(-5, 5), continuous.Function15()),
    ]

    max_iterations = 20
    # brent = BrentMomo(max_iterations)
    brent = BrentNumericalRecipes(max_iterations)

    for optimizer_params, oracle in test_samples:
        test_optimize(brent, optimizer_params, oracle)


if __name__ == "__main__":
    main()
