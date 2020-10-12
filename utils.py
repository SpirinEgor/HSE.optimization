def linear_approximation(x1: float, y1: float, x2: float, y2: float) -> float:
    k = (y2 - y1) / (x2 - x1)
    b = y2 - k * x2
    x3 = -b / k
    return x3
