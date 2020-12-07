from typing import Dict

from assignment_2_3.config import Config
from assignment_2_3.line_searches import (
    GoldenLineSearch,
    BrentLineSearch,
    ArmijoLineSearch,
    WolfeLineSearch,
    NesterovLineSearch,
    AbstractLineSearch,
)


def make_line_search(name: str, config: Config) -> AbstractLineSearch:
    line_searches: Dict[str, AbstractLineSearch.__class__] = {
        GoldenLineSearch.name: GoldenLineSearch,
        BrentLineSearch.name: BrentLineSearch,
        ArmijoLineSearch.name: ArmijoLineSearch,
        WolfeLineSearch.name: WolfeLineSearch,
        NesterovLineSearch.name: NesterovLineSearch,
    }
    if name not in line_searches:
        raise ValueError(f"Unknown line search algorithm '{name}'")
    return line_searches[name](config)
