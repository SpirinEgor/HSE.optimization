from typing import Callable

import numpy


def calculate_gradient(function: Callable, x: numpy.ndarray, eps: float = 1e-8) -> numpy.ndarray:
    """Calculate gradient for given function in given point. Assume function's input is vector.

    :param function: callable function
    :param x: input vector
    :param eps: step in one direction
    :return: gradient vector
    """
    if x.shape[1] != 1:
        raise ValueError("Input x is not a vector")
    gradient = numpy.zeros_like(x)
    step = numpy.zeros_like(x)
    for dim in range(x.shape[0]):
        step[dim] = eps
        gradient[dim] = (function(x + step) - function(x - step)) / (2 * eps)
        step[dim] = 0
    return gradient


def calculate_hessian(gradient: Callable, x: numpy.ndarray, eps: float = 1e-8) -> numpy.ndarray:
    """Calculate hessian of function in given point with knowing gradient function.
    Assume function's input is gradient.

    :param gradient: callable gradient of function
    :param x: input vector
    :param eps: step in one direction
    :return: hessian matrix
    """
    if x.shape[1] != 1:
        raise ValueError("Input x is not a vector")
    hessian = numpy.zeros((x.shape[0], x.shape[0]))
    step = numpy.zeros_like(x)
    for dim in range(x.shape[0]):
        step[dim] = eps
        hessian[dim, :] = (gradient(x + step) - gradient(x - step)).ravel() / (2 * eps)
        step[dim] = 0
    return hessian
