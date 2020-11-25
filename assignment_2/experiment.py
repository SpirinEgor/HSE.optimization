from time import time
from typing import List, Dict, Tuple, Iterable

import numpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from assignment_2.config import COLORS
from assignment_2.line_searches import AbstractLineSearch
from assignment_2.optimize import gradient_descent_optimization, OptimizationStep
from assignment_2.oracles import AbstractOracle


def moving_average(array: Iterable[float], window: int = 25) -> numpy.ndarray:
    cum_sums = numpy.cumsum(array)
    cum_sums[window:] = cum_sums[window:] - cum_sums[:-window]
    return cum_sums[window - 1 :] / window


def experiment_graphic_builder(
    results: Dict[str, List[OptimizationStep]], name: str, true_minimum: float, smooth: bool = False
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=2,
        column_titles=[r"$\log(|F(w_k) - F(w_*)|)$", r"$\log(\frac{|\nabla F(w_k)|^2}{|\nabla F(w_0)|^2})$"],
        horizontal_spacing=0.05,
    )
    figure.update_layout(title=name, height=1000, legend={"orientation": "h"})
    row_names = ["time (ms)", "# oracle calls", "# iterations"]
    for row_id in range(3):
        for col_id in range(2):
            figure.update_yaxes(exponentformat="e", row=row_id + 1, col=col_id + 1)
            figure.update_xaxes(title=row_names[row_id % 3], row=row_id + 1, col=col_id + 1)

    color_iter = iter(COLORS)
    for run, result in results.items():
        wrt_minimum = [numpy.abs(true_minimum - res.value) for res in result]
        wrt_start = [res.stop_criterion for res in result]
        times = [res.passed_time for res in result]
        oracle_calls = [res.oracle_calls for res in result]
        iterations = [i for i, _ in enumerate(result)]

        color = next(color_iter)
        for row_id, x_values in enumerate([times, oracle_calls, iterations]):
            for col_id, y_values in enumerate([wrt_minimum, wrt_start]):
                if col_id == 1 and smooth:
                    y_values = moving_average(y_values)
                y_values = numpy.log(y_values)
                show_legend = row_id == 0 and col_id == 0
                scatter = go.Scatter(
                    x=x_values, y=y_values, name=run, legendgroup=run, line_color=color, showlegend=show_legend
                )
                figure.add_trace(scatter, row=row_id + 1, col=col_id + 1)

    return figure


def gradient_descent_experiment(
    oracle: AbstractOracle,
    line_searches: List[Tuple[str, AbstractLineSearch]],
    start_point: numpy.ndarray,
    tol: float,
    max_iter: int,
) -> Dict[str, List[OptimizationStep]]:
    results = {}
    for name, line_search in line_searches:
        print(f"Run gradient descent optimization for {name}...")
        start_time = time()
        results[name] = gradient_descent_optimization(oracle, line_search, start_point, tol, max_iter)
        print(f"Done in {time() - start_time} ms")

    return results
