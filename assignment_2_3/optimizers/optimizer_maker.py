from typing import Dict

from assignment_2_3.config import Config
from assignment_2_3.optimizers import (
    AbstractOptimizer,
    GradientDescentOptimizer,
    NewtonCholeskyOptimizer,
    HessianFreeNewtonOptimizer,
    LBFGSOptimizer,
)


def make_optimizer(name: str, config: Config) -> AbstractOptimizer:
    known_optimizations: Dict[str, AbstractOptimizer.__class__] = {
        GradientDescentOptimizer.name: GradientDescentOptimizer,
        NewtonCholeskyOptimizer.name: NewtonCholeskyOptimizer,
        HessianFreeNewtonOptimizer.name: HessianFreeNewtonOptimizer,
        LBFGSOptimizer.name: LBFGSOptimizer,
    }
    if name not in known_optimizations:
        raise ValueError(f"Unknown optimizer: {name}")
    return known_optimizations[name](config)
