from time import time
from typing import List, Dict, Iterable

import numpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from assignment_2_3_4.config import COLORS
from assignment_2_3_4.line_searches import AbstractLineSearch
from assignment_2_3_4.optimizers import AbstractOptimizer, OptimizationStep
from assignment_2_3_4.oracles import AbstractOracle


def moving_average(array: Iterable[float], window: int = 25) -> numpy.ndarray:
    cum_sums = numpy.cumsum(array)
    cum_sums[window:] = cum_sums[window:] - cum_sums[:-window]
    return cum_sums[window - 1 :] / window


def experiment_graphic_builder(
    results: Dict[str, List[OptimizationStep]],
    name: str,
    true_minimum: float,
    smooth: bool = False,
    stop_criteria_formula: str = r"$\log(\frac{|\nabla F(w_k)|^2}{|\nabla F(w_0)|^2})$",
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=2,
        column_titles=[r"$\log(|F(w_k) - F(w_*)|)$", stop_criteria_formula],
        horizontal_spacing=0.05,
    )
    figure.update_layout(title=name, height=1000, legend={"orientation": "h"})
    row_names = ["time (ms)", "# oracle calls", "# iterations"]
    for row_id in range(3):
        for col_id in range(2):
            figure.update_yaxes(exponentformat="e", row=row_id + 1, col=col_id + 1, type="log")
            figure.update_xaxes(title=row_names[row_id % 3], row=row_id + 1, col=col_id + 1)

    color_iter = iter(COLORS)
    for run, result in results.items():
        wrt_minimum = [numpy.abs(true_minimum - res.value) + 1e-15 for res in result]
        wrt_start = [res.stop_criterion + 1e-15 for res in result]
        times = [res.passed_time for res in result]
        oracle_calls = [res.oracle_calls for res in result]
        iterations = [i for i, _ in enumerate(result)]

        color = next(color_iter)
        for row_id, x_values in enumerate([times, oracle_calls, iterations]):
            for col_id, y_values in enumerate([wrt_minimum, wrt_start]):
                if col_id == 1 and smooth:
                    y_values = moving_average(y_values)
                show_legend = row_id == 0 and col_id == 0
                scatter = go.Scatter(
                    x=x_values, y=y_values, name=run, legendgroup=run, line_color=color, showlegend=show_legend
                )
                figure.add_trace(scatter, row=row_id + 1, col=col_id + 1)

    return figure


def experiment_runner(
    oracle: AbstractOracle,
    line_searches: List[AbstractLineSearch],
    optimizer: AbstractOptimizer,
    start_point: numpy.ndarray,
) -> Dict[str, List[OptimizationStep]]:
    results = {}
    for line_search in line_searches:
        print(f"Run {optimizer.name} optimization with {line_search.name} line search...")
        start_time = time()
        results[line_search.name] = optimizer.optimize(oracle, start_point, line_search)
        print(f"Done in {time() - start_time} ms")
    return results


def build_lasso_report(
    dataset_lambdas: numpy.ndarray,
    optim_res: Dict[str, List[OptimizationStep]],
    gd_optim_res: List[OptimizationStep],
    oracle: AbstractOracle,
    show_first_steps: int = None,
) -> go.Figure:
    # Create figure with subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
        subplot_titles=[
            "Time to lambda dependency",
            "Number of iterations to lambda dependency",
            "Weights values to lambda dependency",
            "Stop criteria to number of iterations dependency",
            "Loss value to number of iterations dependency",
        ],
        vertical_spacing=0.08,
    )

    # Update x axes properties
    fig.update_xaxes(title=r"$\lambda$", row=1, col=1)
    fig.update_xaxes(title=r"$\lambda$", row=1, col=2)
    fig.update_xaxes(title=r"$\lambda$", type="log", exponentformat="e", row=2, col=1)
    fig.update_xaxes(title="Number of iterations", row=2, col=2)
    fig.update_xaxes(title="Number of iterations", row=3, col=1)

    # Update y axes properties
    fig.update_yaxes(title="Time (s)", type="log", exponentformat="e", row=1, col=1)
    fig.update_yaxes(title="Number of iterations", type="log", exponentformat="e", row=1, col=2)
    fig.update_yaxes(title="Number of zero weights", row=2, col=1)
    fig.update_yaxes(
        title=r"$||\frac{x_k - Prox_{\alpha_k}(x_k - \alpha_k \nabla f(x_k))}{\alpha_k}||^2_2$",
        type="log",
        exponentformat="e",
        row=2,
        col=2,
    )
    fig.update_yaxes(title="Loss", row=3, col=1)

    times = [optim_res[_l][-1].passed_time for _l in dataset_lambdas]
    fig.add_scatter(x=dataset_lambdas, y=times, showlegend=False, row=1, col=1)

    n_iters = [len(optim_res[_l]) for _l in dataset_lambdas]
    fig.add_scatter(x=dataset_lambdas, y=n_iters, showlegend=False, row=1, col=2)

    zero_weights = [(optim_res[_l][-1].point != 0).sum() for _l in dataset_lambdas]
    fig.add_scatter(x=dataset_lambdas, y=zero_weights, mode="lines", showlegend=False, row=2, col=1)

    color_iter = iter(COLORS)
    for lasso_lambda in dataset_lambdas:
        stop_crit = [step.stop_criterion for step in optim_res[lasso_lambda]]
        losses = [
            oracle.value(step.point) + lasso_lambda * numpy.linalg.norm(step.point, ord=1)
            for step in optim_res[lasso_lambda]
        ]
        name = f"$\lambda={lasso_lambda:.2E}$"
        color = next(color_iter)
        fig.add_scatter(
            y=stop_crit[:show_first_steps],
            name=name,
            legendgroup=name,
            showlegend=False,
            line_color=color,
            row=2,
            col=2,
        )
        fig.add_scatter(
            y=losses[:show_first_steps], name=name, legendgroup=name, showlegend=True, line_color=color, row=3, col=1
        )
    gd_losses = [oracle.value(step.point) for step in gd_optim_res]
    fig.add_scatter(y=gd_losses[:show_first_steps], name=f"gradient descent", line_color=next(color_iter), row=3, col=1)

    fig.update_layout(height=1500, legend={"orientation": "h", "y": 1.06})
    return fig
