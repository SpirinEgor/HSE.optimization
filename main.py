import numpy

import oracle
from optimize import Brent


def main():
    true_x_min = 0
    left_bound, right_bound = -10, 10
    epsilon = 1e-8
    max_iterations = 100_000

    oracle_function = oracle.unimodal.quadratic(true_x_min)
    oracle_wrapper = oracle.OracleWrapper(oracle_function)

    brent = Brent(max_iterations)
    x_min = brent.brent_with_derivatives(oracle_wrapper, left_bound, right_bound, epsilon)
    print(f"Optimized minimum is {x_min}, true minimum is {true_x_min} (error: {numpy.abs(x_min - true_x_min)})\n"
          f"Number of calls to the oracle is {oracle_wrapper.get_number_of_calls()}")


if __name__ == "__main__":
    main()
