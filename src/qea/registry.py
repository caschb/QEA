"""Explicit registration of algorithms and their parameter schemas."""

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, cast

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
    sections: dict,
    names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    unknown = sorted(set(sections) - ALGORITHMS.keys())
    if unknown:
        msg = f"Unknown algorithms: {', '.join(unknown)}"
        raise ConfigError(msg)
    parameters = {}
    requested = names if names is not None else ("qea", "ga")
    for name in dict.fromkeys((*requested, *sections)):
        if name not in ALGORITHMS:
            msg = f"Unknown algorithm: {name}"
            raise ConfigError(msg)
        definition = ALGORITHMS[name]
        section = sections.get(name, {})
        if not isinstance(section, dict):
            msg = f"algorithms.{name} must be a TOML table"
            raise ConfigError(msg)
        allowed = {f.name for f in fields(cast("Any", definition.parameters))}
        unknown = sorted(set(section) - allowed)
        if unknown:
            msg = f"Unknown keys in algorithms.{name}: {', '.join(unknown)}"
            raise ConfigError(msg)
        try:
            parameters[name] = definition.parameters(**section)
        except ConfigError as err:
            msg = f"algorithms.{name}: {err}"
            raise ConfigError(msg) from err
    return parameters


def validate_parameters(name: str, parameters: object) -> None:
    if name not in ALGORITHMS:
        msg = f"Unknown algorithm: {name}"
        raise ConfigError(msg)
    definition = ALGORITHMS[name]
    if not isinstance(parameters, definition.parameters):
        msg = f"Invalid parameters for algorithm {name}"
        raise ConfigError(msg)


def create_algorithm(name: str, parameters: object) -> EvolutionaryAlgorithm:
    validate_parameters(name, parameters)
    return ALGORITHMS[name].factory(parameters)
