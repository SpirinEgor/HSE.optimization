from abc import ABC, abstractmethod

import numpy


def gen_pos_def_matrix(n: int) -> numpy.ndarray:
    sqrt_matrix = numpy.random.randn(n, n)
    return sqrt_matrix @ sqrt_matrix.T


class AbstractOracle(ABC):
    _n: int = None

    @abstractmethod
    def value(self, x: numpy.ndarray) -> float:
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()

    @property
    def n(self):
        return self._n


class Oracle41(AbstractOracle):
    def __init__(self, n: int):
        self._n = n
        self._A = gen_pos_def_matrix(n)

    def value(self, x: numpy.ndarray) -> float:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return 0.5 * numpy.linalg.norm(x @ x.T - self._A) ** 2

    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return 2 * (x @ x.T @ x - self._A @ x)

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        i_n = numpy.identity(self._n)
        return 2 * (x.T @ x) * i_n + 4 * (x @ x.T) - 2 * self._A


class Oracle42(AbstractOracle):
    def __init__(self, n: int):
        self._n = n
        self._A = gen_pos_def_matrix(n)

    def value(self, x: numpy.ndarray) -> float:
        return x.T @ (self._A @ x) / numpy.linalg.norm(x) ** 2

    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        x_norm = numpy.linalg.norm(x)
        return 2 * ((self._A @ x) * x_norm ** 2 - (x @ x.T @ self._A @ x)) / x_norm ** 4

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        x_norm = numpy.linalg.norm(x)
        i_n = numpy.identity(self._n)

        return (
            2
            / x_norm ** 4
            * (
                x_norm ** 2 * self._A
                - 2 * self._A @ x @ x.T
                - x.T @ self._A @ x * i_n
                - 2 * x @ x.T @ self._A
                + (4 / x_norm ** 2) * x @ x.T @ self._A @ x @ x.T
            )
        )


class Oracle43(AbstractOracle):
    def __init__(self, n: int):
        self._n = n

    def value(self, x: numpy.ndarray) -> float:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        scalar = x.T @ x
        return numpy.power(scalar, scalar)

    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        scalar = x.T @ x
        return 2 * numpy.power(scalar, scalar) * (numpy.log(scalar) + 1) * x

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        i_n = numpy.identity(self._n)
        scalar = x.T @ x
        return (
            2
            * numpy.power(scalar, scalar)
            * (2 * ((numpy.log(scalar) + 1) ** 2 + 1 / scalar) * x @ x.T + (numpy.log(scalar) + 1) * i_n)
        )


class Oracle61(AbstractOracle):
    _n: int = 2

    def value(self, x: numpy.ndarray) -> float:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return 2 * x[0] ** 2 + x[1] ** 2 * (x[0] ** 2 - 2)

    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return numpy.array([4 * x[0] + 2 * x[1] ** 2 * x[0], 2 * x[0] ** 2 * x[1] - 4 * x[1]])

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        hessian = numpy.zeros((2, 2))
        hessian[0, 0] = 4 + 2 * x[1] ** 2
        hessian[0, 1] = hessian[1, 0] = 4 * x[0] * x[1]
        hessian[1, 1] = 2 * x[0] ** 2 - 4
        return hessian


class Oracle62(AbstractOracle):
    _n: int = 2

    def __init__(self, lam: float):
        self._lambda = lam

    def value(self, x: numpy.ndarray) -> float:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return (1 - x[0]) ** 2 + self._lambda * (x[1] - x[0] ** 2) ** 2

    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return numpy.array(
            [-2 * (1 - x[0] + 2 * self._lambda * x[0] * (x[1] - x[0] ** 2)), 2 * self._lambda * (x[1] - x[0] ** 2)]
        )

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        hessian = numpy.zeros((2, 2))
        hessian[0, 0] = -2 * (2 * self._lambda * x[1] - 6 * self._lambda * x[0] ** 2 - 1)
        hessian[0, 1] = hessian[1, 0] = -4 * self._lambda * x[0]
        hessian[1, 1] = 2 * self._lambda
        return hessian


class Oracle71(AbstractOracle):
    function: str = r"$x^3$"
    _n: int = 1

    def value(self, x: numpy.ndarray) -> float:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return x ** 3

    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return 3 * x ** 2

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        pass


class Oracle72(AbstractOracle):
    function: str = r"$\sum \log(x_i + 1)$"

    def __init__(self, n: int):
        self._n = n

    def value(self, x: numpy.ndarray) -> float:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return (numpy.log(x + 1)).sum()

    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return 1.0 / (x + 1)

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        pass


class Oracle73(AbstractOracle):
    function: str = r"$e^x$"
    _n: int = 1

    def value(self, x: numpy.ndarray) -> float:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return numpy.exp(x)

    def gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        if x.shape[0] != self._n:
            raise ValueError(f"X has error shape, should be ({self._n},)")
        return numpy.exp(x)

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        pass
