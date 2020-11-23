from typing import Callable, Tuple

from numpy import abs, sign

from assignment_1.optimize import OptimizeResult
from assignment_1.optimize.brent import IBrent


class BrentNumericalRecipes(IBrent):
    """https://e-maxx.ru/bookz/files/numerical_recipes.pdf (501)
    """

    _min_tol = 1.0e-10

    def brent_with_derivatives(
            self, oracle: Callable[[float], Tuple[float, float]], a: float, c: float, eps: float
    ) -> OptimizeResult:
        _history = []

        # Init block
        a, c = (a, c) if a < c else (c, a)
        current_step = previous_step = c - a
        init_xs, init_f_xs, init_df_xs = self._init_brent_variables(oracle, a, c)
        first_min, second_min, third_min = init_xs
        f_first_min, f_second_min, f_third_min = init_f_xs
        df_first_min, df_second_min, df_third_min = init_df_xs

        for n_iter in range(self._max_iterations):
            _history.append(first_min)

            median = 0.5 * (a + c)
            tol1 = eps * abs(first_min) + self._min_tol
            tol2 = 2 * tol1
            if abs(first_min - median) <= tol2 - 0.5 * (c - a) or abs(df_first_min) <= self._min_tol:
                return OptimizeResult(first_min, _history, n_iter)

            temp_step = previous_step
            previous_step = current_step
            # secant method
            if abs(previous_step) > tol1:
                d1 = d2 = 2 * (c - a)
                if df_first_min != df_second_min:
                    d1 = (second_min - first_min) * df_first_min / (df_first_min - df_second_min)
                if df_first_min != df_third_min:
                    d2 = (third_min - first_min) * df_first_min / (df_first_min - df_third_min)
                u1 = first_min + d1
                u2 = first_min + d2
                is_ok1 = (a - u1) * (u1 - c) > 0 >= df_first_min * d1
                is_ok2 = (a - u2) * (u2 - c) > 0 >= df_first_min * d2

                if is_ok1 or is_ok2:
                    if is_ok1 and is_ok2:
                        current_step = d1 if abs(d1) < abs(d2) else d2
                    else:
                        current_step = d1 if is_ok1 else d2

                    if abs(current_step) <= abs(0.5 * temp_step):
                        next_min = first_min + current_step
                        if next_min - a < tol2 or c - next_min < tol2:
                            current_step = sign(median - first_min) * tol1
                    else:
                        current_step = (a - first_min) / 2 if df_first_min >= 0 else (c - first_min) / 2
                else:
                    current_step = (a - first_min) / 2 if df_first_min >= 0 else (c - first_min) / 2

            # bisect method
            else:
                current_step = (a - first_min) / 2 if df_first_min >= 0 else (c - first_min) / 2

            if abs(current_step) >= tol1:
                next_min = first_min + current_step
                f_next_min, df_next_min = oracle(next_min)
            else:
                next_min = first_min + sign(current_step) * tol1
                f_next_min, df_next_min = oracle(next_min)
                if f_next_min > f_first_min:
                    return OptimizeResult(first_min, _history, n_iter)

            # update brackets and points
            if f_next_min <= f_first_min:
                a, c = (first_min, c) if next_min >= first_min else (a, first_min)

                third_min, f_third_min, df_third_min = second_min, f_second_min, df_second_min
                second_min, f_second_min, df_second_min = first_min, f_first_min, df_first_min
                first_min, f_first_min, df_first_min = next_min, f_next_min, df_next_min
            else:
                a, c = (a, next_min) if next_min >= first_min else (next_min, c)
                if f_next_min <= f_second_min or second_min == first_min:
                    third_min, f_third_min, df_third_min = second_min, f_second_min, df_second_min
                    second_min, f_second_min, df_second_min = next_min, f_next_min, df_next_min
                elif f_next_min <= f_third_min or third_min == first_min or third_min == second_min:
                    third_min, f_third_min, df_third_min = next_min, f_next_min, df_next_min

        return OptimizeResult(first_min, _history, self._max_iterations)
