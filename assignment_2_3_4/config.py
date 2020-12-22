from dataclasses import dataclass


@dataclass
class Config:
    data_folder: str = "../data"

    tol: float = 1e-8
    max_iter: int = 10_000

    bracket_left: int = 0
    bracket_right: int = 10
    max_iter_line_search: int = 100
    armijo_c: float = 0.5
    nesterov_c: float = 0.5
    wolfe_second_c: float = 0.9

    l_bfgs_history_size: int = 100

    lasso_lambda: float = 0.5
    lasso_start_l: float = 1.0


# https://community.plotly.com/t/plotly-colours-list/11730
COLORS = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]
