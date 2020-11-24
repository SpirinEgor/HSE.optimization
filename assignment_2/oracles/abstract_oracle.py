from abc import ABC
from typing import Tuple

import numpy


class AbstractOracle(ABC):

    def __init__(self, x: numpy.ndarray, y: numpy.ndarray):
        self._x = x
        self._y = y.reshape(-1, 1)
        self._n_samples, self._n_features = self._x.shape

    @staticmethod
    def _save_weights_use(weights: numpy.ndarray) -> numpy.ndarray:
        if len(weights.shape) == 1:
            return weights.reshape(-1, 1)
        return weights

    def value(self, weights: numpy.ndarray) -> numpy.float:
        pass

    def grad(self, weights: numpy.ndarray) -> numpy.ndarray:
        pass

    def hessian(self, weights: numpy.ndarray) -> numpy.ndarray:
        pass

    def hessian_vec_product(self, weights: numpy.ndarray, d: numpy.ndarray) -> numpy.ndarray:
        pass

    def fuse_value_grad(self, weights: numpy.ndarray) -> Tuple[numpy.float, numpy.ndarray]:
        return self.value(weights), self.grad(weights)

    def fuse_value_grad_hessian(self, weights: numpy.ndarray) -> Tuple[numpy.float, numpy.ndarray, numpy.ndarray]:
        return self.value(weights), self.grad(weights), self.hessian(weights)

    def fuse_value_grad_hessian_vec_product(
            self, weights: numpy.ndarray, d: numpy.ndarray
    ) -> Tuple[numpy.float, numpy.ndarray, numpy.ndarray]:
        return self.value(weights), self.grad(weights), self.hessian_vec_product(weights, d)
