from typing import Callable, Tuple, Dict

import numpy

from assignment_1.optimize import BrentNumericalRecipes, OptimizeResult


# Требуется реализовать метод: который будет находить минимум функции на отрезке [a,b]
def optimize(
    oracle: Callable[[float], Tuple[float, float]],
    a: float,
    b: float,
    eps: float = 1e-8,
    optimizer_params: Dict = None,
) -> numpy.ndarray:
    if optimizer_params is None:
        optimizer_params = {}
    optimize_function = BrentNumericalRecipes(**optimizer_params).get_optimize_function()
    optimize_result: OptimizeResult = optimize_function(oracle, a, b, eps)
    return numpy.array(optimize_result.x_min)


# Задание состоит из 2-х частей — реализовать любой алгоритм оптимизации по выбору
# Провести анализ работы алгоритма на нескольких функция, построить графики сходимости вида:
# кол-во итераций vs log(точность); время работы vs log(точность)
# Изучить, как метод будет работать на неунимодальных функций и привести примеры, подтверждающие поведение
# (например, что будет сходится в ближайший локальный минимум)


# Критерий оценки:
# 4-5 баллов — решение работает и дает правильный ответ,
# код реализации не вызывает вопрос + ipynb отчет с исследованием работы метода

# Оценка по дальнейшим результатам: будет 4-5 тестовых функций.
# На каждой будет для всех сданных решений строится распределение времени работы
# Далее по квантилям распределения: 10: 95%, 9: 85%, 8: 75%, 7: 50% — по каждому заданию независимо,
# далее среднее по всем
# Дополнительно требование на 8+ баллов: минимальное требование обогнать бейзлайн-решение
# (скрыт от вас, простая наивная реализация одного из методов с лекции)
