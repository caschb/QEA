"""Experiment settings and sectioned TOML loading."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import create_algorithm, default_parameters, load_parameters
from .validation import (
    MIN_AGENTS,
    MIN_REPLICAS,
    SCENARIOS,
    ConfigError,
    _check_choice,
    _check_constraints,
    _check_int,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared problem/run settings plus separately validated algorithm parameters."""

    n_agents: int = 8
    scenario: str = "nominal"
    max_generations: int = 150
    seed: int = 42
    constraints: list = field(default_factory=list)
    n_replicas: int = 8
    algorithms: dict[str, Any] = field(default_factory=default_parameters)

    def __post_init__(self) -> None:
        _check_int("n_agents", self.n_agents, MIN_AGENTS)
        _check_choice("scenario", self.scenario, SCENARIOS)
        _check_int("max_generations", self.max_generations, 1)
        _check_int("seed", self.seed, 0)
        _check_constraints(self.constraints, self.n_agents)
        _check_int("n_replicas", self.n_replicas, MIN_REPLICAS)
        if not isinstance(self.algorithms, dict):
            raise ConfigError("algorithms must be a mapping of validated parameters")
        for name, parameters in self.algorithms.items():
            create_algorithm(name, parameters)
        if not {"qea", "ga"} <= self.algorithms.keys():
            raise ConfigError("The current comparison requires qea and ga parameters")


def _section(data: dict, name: str, allowed: set[str]) -> dict:
    section = data.get(name, {})
    if not isinstance(section, dict):
        raise ConfigError(f"{name} must be a TOML table")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys in {name}: {', '.join(unknown)}")
    return section


def read_config(config_path: Path | None) -> ExperimentConfig:
    if config_path is None:
        return ExperimentConfig()
    try:
        with config_path.open(mode="rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise ConfigError(
            f"Could not read TOML configuration {config_path}: {err}"
        ) from err
    _section({"root": data}, "root", {"problem", "experiment", "algorithms"})
    problem = _section(data, "problem", {"n_agents", "scenario", "constraints"})
    experiment = _section(data, "experiment", {"seed", "max_generations", "n_replicas"})
    algorithms = data.get("algorithms", {})
    if not isinstance(algorithms, dict):
        raise ConfigError("algorithms must be a TOML table")
    return ExperimentConfig(
        **problem, **experiment, algorithms=load_parameters(algorithms)
    )
