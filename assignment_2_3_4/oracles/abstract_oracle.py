from abc import ABC
from typing import Tuple, Union

import numpy
from scipy.sparse import csr_matrix

Matrix = Union[numpy.ndarray, csr_matrix]


class AbstractOracle(ABC):
    _call_counter = 0
    name: str = None

    def __init__(self, x: Matrix, y: numpy.ndarray):
        self._x = x
        self._y = y
        self._n_samples, self._n_features = self._x.shape

    def get_true_minimum(self, tol: float, max_iter: float) -> float:
        raise NotImplementedError()

    # ========== Properties ===========

    @property
    def n_calls(self) -> int:
        return self._call_counter

    def reset_call_counter(self):
        self._call_counter = 0

    @property
    def n_features(self) -> int:
        return self._n_features

    # ========== Oracle Interface ==========

    def value(self, weights: numpy.ndarray) -> float:
        raise NotImplementedError()

    def grad(self, weights: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()

    def hessian(self, weights: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()

    def hessian_vec_product(self, weights: numpy.ndarray, d: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()

    def fuse_value_grad(self, weights: numpy.ndarray) -> Tuple[float, numpy.ndarray]:
        raise NotImplementedError()

    def fuse_value_grad_hessian(self, weights: numpy.ndarray) -> Tuple[float, numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError()

    def fuse_value_grad_hessian_vec_product(
        self, weights: numpy.ndarray, d: numpy.ndarray
    ) -> Tuple[float, numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError()
