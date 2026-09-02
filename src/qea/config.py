import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ConfigError(Exception): ...


@dataclass(frozen=True)
class ExperimentConfig:
    """Parámetros únicos compartidos por QEA y GA."""

    n_agents: int = 8
    scenario: str = "nominal"
    max_generations: int = 150
    seed: int = 42
    # Restricciones de organización (node_list para set_constraint_org_team)
    # Ejemplo: [[1,2,3],[1,4,5]]  →  nodo 1 es maestro de 2,3 y de 4,5
    constraints: list = field(default_factory=list)
    # QEA
    theta_initial: float = 0.05 * np.pi
    theta_min: float = 0.001 * np.pi
    decay_rate: float = 0.02
    rotation_scheme: str = "I"  # "I" | "II" | "III"  — ver QuantumChromosome
    use_qiskit: bool = True
    # GA
    pop_size: int = 30
    mutation_rate: float = 0.02
    crossover_rate: float = 0.8

    def __post_init__(self) -> None:
        if self.n_agents < 0:
            msg = "The number of agents must be a positive value"
            raise ConfigError(msg)


def read_config(config_path: Path) -> ExperimentConfig:
    with Path.open(config_path, mode="rb") as f:
        data = tomllib.load(f)

    return ExperimentConfig(**data)
