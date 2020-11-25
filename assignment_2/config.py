from dataclasses import dataclass


@dataclass
class Config:
    data_folder = "../data"

    tol = 1e-8
    max_iter = 10_000

    bracket_left = 0
    bracket_right = 100
    iter_line_search = 1000
    armijo_c = 0.5
    nesterov_c = 0.5
    wolfe_second_c = 0.9


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
