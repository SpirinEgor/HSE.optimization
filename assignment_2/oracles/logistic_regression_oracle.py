import numpy
from scipy.special import expit

from assignment_2.oracles import AbstractOracle


class LogisticRegressionOracle(AbstractOracle):
    _eps = 1e-10

    def __init__(self, x: numpy.ndarray, y: numpy.ndarray):
        super().__init__(x, y)
        self._y_t = self._y.T
        self._x_t = self._x.T

    def value(self, weights: numpy.ndarray) -> numpy.float:
        weights = self._save_weights_use(weights)

        z = self._x @ weights
        logit = expit(z)
        loss = -(
                self._y_t @ numpy.log(logit + self._eps) +
                (1 - self._y_t) @ numpy.log(1 - logit + self._eps)
        ) / self._n_samples
        loss = loss[0, 0]

        assert isinstance(loss, numpy.float)
        return loss

    def grad(self, weights: numpy.ndarray) -> numpy.ndarray:
        weights = self._save_weights_use(weights)

        logit = expit(self._x @ weights)
        result = self._x_t @ (logit - self._y) / self._n_samples

        assert weights.shape == result.shape
        return result

    def hessian(self, weights: numpy.ndarray) -> numpy.ndarray:
        weights = self._save_weights_use(weights)

        logit = expit(self._x @ weights)[:, 0]
        diag_term = numpy.diag(logit * (1 - logit))
        result = self._x_t @ diag_term @ self._x / self._n_samples

        correct_shape = (weights.shape[0], weights.shape[0])
        assert result.shape == correct_shape
        return result

    def hessian_vec_product(self, weights: numpy.ndarray, d: numpy.ndarray) -> numpy.ndarray:
        d = self._save_weights_use(d)
        hessian = self.hessian(weights)
        return hessian.dot(d)
