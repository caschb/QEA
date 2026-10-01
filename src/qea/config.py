"""Experiment settings and sectioned TOML loading."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import default_parameters, load_parameters, validate_parameters
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
    selected_algorithms: tuple[str, ...] = ("qea", "ga")
    algorithms: dict[str, Any] = field(default_factory=default_parameters)

    def __post_init__(self) -> None:
        _check_int("n_agents", self.n_agents, MIN_AGENTS)
        _check_choice("scenario", self.scenario, SCENARIOS)
        _check_int("max_generations", self.max_generations, 1)
        _check_int("seed", self.seed, 0)
        _check_constraints(self.constraints, self.n_agents)
        _check_int("n_replicas", self.n_replicas, MIN_REPLICAS)
        if not isinstance(self.algorithms, dict):
            msg = "algorithms must be a mapping of validated parameters"
            raise ConfigError(msg)
        for name, parameters in self.algorithms.items():
            validate_parameters(name, parameters)
        validate_selection(self.selected_algorithms)
        missing = set(self.selected_algorithms) - self.algorithms.keys()
        if missing:
            msg = f"Missing parameters for algorithms: {', '.join(sorted(missing))}"
            raise ConfigError(msg)


def validate_selection(names: object) -> None:
    if not isinstance(names, (list, tuple)) or not names:
        msg = "experiment.algorithms must be a nonempty list of names"
        raise ConfigError(msg)
    if any(not isinstance(name, str) for name in names):
        msg = "experiment.algorithms must contain string names"
        raise ConfigError(msg)
    if len(set(names)) != len(names):
        msg = "experiment.algorithms must not repeat a name"
        raise ConfigError(msg)


def _section(data: dict, name: str, allowed: set[str]) -> dict:
    section = data.get(name, {})
    if not isinstance(section, dict):
        msg = f"{name} must be a TOML table"
        raise ConfigError(msg)
    unknown = sorted(set(section) - allowed)
    if unknown:
        msg = f"Unknown keys in {name}: {', '.join(unknown)}"
        raise ConfigError(msg)
    return section


def read_config(config_path: Path | None) -> ExperimentConfig:
    if config_path is None:
        return ExperimentConfig()
    try:
        with config_path.open(mode="rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as err:
        msg = f"Could not read TOML configuration {config_path}: {err}"
        raise ConfigError(msg) from err
    _section({"root": data}, "root", {"problem", "experiment", "algorithms"})
    problem = _section(data, "problem", {"n_agents", "scenario", "constraints"})
    experiment = _section(
        data,
        "experiment",
        {"seed", "max_generations", "n_replicas", "algorithms"},
    )
    algorithms = data.get("algorithms", {})
    if not isinstance(algorithms, dict):
        msg = "algorithms must be a TOML table"
        raise ConfigError(msg)
    selected = experiment.pop("algorithms", ["qea", "ga"])
    validate_selection(selected)
    return ExperimentConfig(
        **problem,
        **experiment,
        selected_algorithms=tuple(selected),
        algorithms=load_parameters(algorithms, tuple(selected)),
    )
