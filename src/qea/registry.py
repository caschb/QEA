"""Explicit registration of algorithms and their parameter schemas."""

from dataclasses import dataclass, fields
from typing import Any, Callable, cast

from .algorithms.ga import GAConfig, GeneticAlgorithm
from .algorithms.qea import QEA, QEAConfig
from .interface import EvolutionaryAlgorithm
from .validation import ConfigError


@dataclass(frozen=True)
class AlgorithmDefinition:
    parameters: type
    factory: Callable[[Any], EvolutionaryAlgorithm]


ALGORITHMS = {
    "qea": AlgorithmDefinition(QEAConfig, QEA),
    "ga": AlgorithmDefinition(GAConfig, GeneticAlgorithm),
}


def default_parameters() -> dict[str, Any]:
    return {
        name: definition.parameters()
        for name, definition in ALGORITHMS.items()
        if name in ("qea", "ga")
    }


def load_parameters(
    sections: dict, names: tuple[str, ...] | None = None
) -> dict[str, Any]:
    unknown = sorted(set(sections) - ALGORITHMS.keys())
    if unknown:
        raise ConfigError(f"Unknown algorithms: {', '.join(unknown)}")
    parameters = {}
    requested = names if names is not None else ("qea", "ga")
    for name in dict.fromkeys((*requested, *sections)):
        if name not in ALGORITHMS:
            raise ConfigError(f"Unknown algorithm: {name}")
        definition = ALGORITHMS[name]
        section = sections.get(name, {})
        if not isinstance(section, dict):
            raise ConfigError(f"algorithms.{name} must be a TOML table")
        allowed = {f.name for f in fields(cast(Any, definition.parameters))}
        unknown = sorted(set(section) - allowed)
        if unknown:
            raise ConfigError(
                f"Unknown keys in algorithms.{name}: {', '.join(unknown)}"
            )
        try:
            parameters[name] = definition.parameters(**section)
        except ConfigError as err:
            raise ConfigError(f"algorithms.{name}: {err}") from err
    return parameters


def validate_parameters(name: str, parameters: object) -> None:
    if name not in ALGORITHMS:
        raise ConfigError(f"Unknown algorithm: {name}")
    definition = ALGORITHMS[name]
    if not isinstance(parameters, definition.parameters):
        raise ConfigError(f"Invalid parameters for algorithm {name}")


def create_algorithm(name: str, parameters: object) -> EvolutionaryAlgorithm:
    validate_parameters(name, parameters)
    return ALGORITHMS[name].factory(parameters)
