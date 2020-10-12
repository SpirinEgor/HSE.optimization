from typing import Callable, Tuple

__all__ = ["quadratic", "example"]


# Пусть дана функция (оракул) f
# Со следующим интерфейсом
# def f(x):
#     return f(x), f'(x)
# Например, оракул для квадратичной фунции x^2/2
def example() -> Callable[[float], Tuple[float, float]]:
    def f(x):
        return x * x / 2, x
    return f


def quadratic(min_x: float) -> Callable[[float], Tuple[float, float]]:
    """return first order oracle for function f(x) = (x - k)^2
    minimum of this function in x=k

    :param min_x: k for function
    :return: function for producing oracle output
    """
    def f(x):
        return (x - min_x) ** 2, 2 * (x - min_x)
    return f
