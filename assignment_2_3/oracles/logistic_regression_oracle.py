from typing import Tuple

import numpy
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from assignment_2_3.oracles import AbstractOracle
from assignment_2_3.oracles.abstract_oracle import Matrix


class LogisticRegressionOracle(AbstractOracle):
    _eps = 1e-10
    name: str = "logistic regression"

    def __init__(self, x: Matrix, y: numpy.ndarray):
        super().__init__(x, y)
        self._x_t = self._x.T

    def get_true_minimum(self, tol: float, max_iter: float) -> float:
        sklearn_logit_reg = LogisticRegression(
            tol=tol, max_iter=max_iter, random_state=7, fit_intercept=False, penalty="none"
        )
        sklearn_logit_reg.fit(self._x, self._y)
        minimum_point = sklearn_logit_reg.coef_[0]
        return self.value(minimum_point)

    # ========== Logistic Regression Calculations ==========

    def _get_loss(self, logit: numpy.ndarray) -> numpy.float:
        return -numpy.mean(numpy.log(numpy.where(self._y != 0, logit + self._eps, 1 - logit + self._eps)))

    def _get_grad(self, logit: numpy.ndarray) -> numpy.ndarray:
        return (logit - self._y) @ self._x / self._n_samples

    def _get_hessian(self, logit: numpy.ndarray) -> numpy.ndarray:
        logit = (logit * (1 - logit)).reshape(-1, 1) / self._n_samples
        if isinstance(self._x_t, numpy.ndarray):
            return self._x_t @ (self._x * logit) / self._n_samples
        else:
            return (self._x_t @ (self._x.multiply(logit))).toarray()

    def _get_hessian_dot_vector(self, logit: numpy.ndarray, d: numpy.ndarray) -> numpy.ndarray:
        logit = logit * (1 - logit) / self._n_samples
        return ((self._x @ d) * logit) @ self._x

    # ========== Oracle Interface ==========

    def value(self, weights: numpy.ndarray) -> numpy.float:
        self._call_counter += 1
        return self._get_loss(expit(self._x.dot(weights)))

    def grad(self, weights: numpy.ndarray) -> numpy.ndarray:
        self._call_counter += 1
        return self._get_grad(expit(self._x.dot(weights)))

    def hessian(self, weights: numpy.ndarray) -> numpy.ndarray:
        self._call_counter += 1
        return self._get_hessian(expit(self._x.dot(weights)))

    def hessian_vec_product(self, weights: numpy.ndarray, d: numpy.ndarray) -> numpy.ndarray:
        self._call_counter += 1
        return self._get_hessian_dot_vector(expit(self._x.dot(weights)), d)

    def fuse_value_grad(self, weights: numpy.ndarray) -> Tuple[numpy.float, numpy.ndarray]:
        self._call_counter += 1
        logit = expit(self._x.dot(weights))
        return self._get_loss(logit), self._get_grad(logit)

    def fuse_value_grad_hessian(self, weights: numpy.ndarray) -> Tuple[numpy.float, numpy.ndarray, numpy.ndarray]:
        self._call_counter += 1
        logit = expit(self._x.dot(weights))
        return self._get_loss(logit), self._get_grad(logit), self._get_hessian(logit)

    def fuse_value_grad_hessian_vec_product(
        self, weights: numpy.ndarray, d: numpy.ndarray
    ) -> Tuple[numpy.float, numpy.ndarray, numpy.ndarray]:
        self._call_counter += 1
        logit = expit(self._x.dot(weights))
        return (
            self._get_loss(logit),
            self._get_grad(logit),
            self._get_hessian_dot_vector(logit, d),
        )
