"""Results returned by the evolutionary algorithms."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AlgorithmResult:
    """Resultado estándar, igual para QEA y GA."""

    best_binary: np.ndarray
    best_fitness: float
    fitness_history: list[float] = field(default_factory=list)
    best_history: list[float] = field(default_factory=list)
    diversity_history: list[float] = field(default_factory=list)
    time_elapsed: float = 0.0
    label: str = ""
