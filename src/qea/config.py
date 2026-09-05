import tomllib
from dataclasses import dataclass, field, fields
from functools import cache
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator


class ConfigError(Exception): ...


SCENARIOS = ("nominal", "safe", "critical")
ROTATION_SCHEMES = ("I", "II", "III")

# El cruce del GA parte el cromosoma en rng.integers(1, n_genes), que exige
# n_genes >= 2, y n_genes = n(n-1)/2, así que hacen falta 3 agentes.
MIN_AGENTS = 3
# _tournament() toma 3 individuos sin reemplazo.
MIN_POP_SIZE = 3
# Un equipo es un maestro y al menos un subordinado.
MIN_TEAM_SIZE = 2
# La prueba de Wilcoxon necesita al menos dos pares para comparar.
MIN_REPLICAS = 2


@cache
def _aer_methods() -> tuple[str, ...]:
    """Métodos de simulación que admite la versión de qiskit-aer instalada.

    Se consulta a Aer en vez de fijar la lista aquí para que no se
    desactualice. El resultado se cachea y solo se pide bajo demanda: una
    corrida clásica (``use_qiskit = false``) nunca construye un AerSimulator.
    """
    return tuple(AerSimulator().available_methods())


def _check_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        msg = f"{name} must be true or false, got {value!r}"
        raise ConfigError(msg)


def _check_choice(name: str, value: object, options: tuple[str, ...]) -> None:
    if value not in options:
        allowed = ", ".join(repr(o) for o in options)
        msg = f"{name} must be one of {allowed}, got {value!r}"
        raise ConfigError(msg)


def _check_int(name: str, value: object, low: int) -> None:
    # bool es subclase de int; `n_agents = true` no es una configuración válida.
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer, got {value!r}"
        raise ConfigError(msg)
    if value < low:
        msg = f"{name} must be >= {low}, got {value}"
        raise ConfigError(msg)


def _check_number(
    name: str,
    value: object,
    low: float,
    high: float | None = None,
    *,
    low_exclusive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a number, got {value!r}"
        raise ConfigError(msg)
    if low_exclusive and value <= low:
        msg = f"{name} must be > {low}, got {value}"
        raise ConfigError(msg)
    if not low_exclusive and value < low:
        msg = f"{name} must be >= {low}, got {value}"
        raise ConfigError(msg)
    if high is not None and value > high:
        msg = f"{name} must be <= {high}, got {value}"
        raise ConfigError(msg)


def _check_constraints(constraints: object, n_agents: int) -> None:
    if not isinstance(constraints, list):
        msg = f"constraints must be a list of teams, got {constraints!r}"
        raise ConfigError(msg)

    for position, team in enumerate(constraints, start=1):
        label = f"constraints[{position}]"
        if not isinstance(team, list):
            msg = f"{label} must be a list of nodes, got {team!r}"
            raise ConfigError(msg)
        # set_constraint_org_team() descarta en silencio los equipos de menos
        # de dos nodos, así que la restricción nunca llegaría a aplicarse.
        if len(team) < MIN_TEAM_SIZE:
            msg = (
                f"{label} needs at least {MIN_TEAM_SIZE} nodes (a master and one "
                f"subordinate), got {team!r}"
            )
            raise ConfigError(msg)
        for node in team:
            if isinstance(node, bool) or not isinstance(node, int):
                msg = f"{label} must contain integer node numbers, got {node!r}"
                raise ConfigError(msg)
            # Los nodos indexan la matriz de genes; fuera de rango corrompe
            # el cromosoma o revienta con un IndexError opaco.
            if not 1 <= node <= n_agents:
                msg = (
                    f"{label} node {node} is out of range: nodes are numbered "
                    f"1 to {n_agents}"
                )
                raise ConfigError(msg)
        if len(set(team)) != len(team):
            msg = f"{label} repeats a node: {team!r}"
            raise ConfigError(msg)


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
    rotation_scheme: str = "I"  # "I" | "II" | "III"  ver QuantumChromosome
    use_qiskit: bool = True
    # Método de simulación de AerSimulator; solo se valida y se usa si
    # use_qiskit. "automatic" deja que Aer elija según el circuito.
    aer_method: str = "automatic"
    # GA
    pop_size: int = 30
    mutation_rate: float = 0.02
    crossover_rate: float = 0.8

    n_replicas: int = 8

    def __post_init__(self) -> None:
        # Problema
        _check_int("n_agents", self.n_agents, MIN_AGENTS)
        _check_choice("scenario", self.scenario, SCENARIOS)
        _check_int("max_generations", self.max_generations, 1)
        _check_int("seed", self.seed, 0)  # np.random.default_rng los exige >= 0
        _check_constraints(self.constraints, self.n_agents)

        # QEA
        _check_number("theta_initial", self.theta_initial, 0.0, low_exclusive=True)
        _check_number("theta_min", self.theta_min, 0.0, low_exclusive=True)
        if self.theta_min > self.theta_initial:
            # _gdaa() devuelve max(theta, theta_min): un piso por encima del
            # ángulo inicial deja el ángulo constante y anula el decaimiento.
            msg = (
                f"theta_min ({self.theta_min}) must not exceed theta_initial "
                f"({self.theta_initial}); the GDAA decay would never apply"
            )
            raise ConfigError(msg)
        _check_number("decay_rate", self.decay_rate, 0.0)
        # apply_rotation() no tiene rama por defecto: un esquema desconocido
        # no rota ningún qubit y degrada el QEA a búsqueda aleatoria en silencio.
        _check_choice("rotation_scheme", self.rotation_scheme, ROTATION_SCHEMES)
        _check_bool("use_qiskit", self.use_qiskit)
        if self.use_qiskit:
            _check_choice("aer_method", self.aer_method, _aer_methods())

        # GA
        _check_int("pop_size", self.pop_size, MIN_POP_SIZE)
        _check_number("mutation_rate", self.mutation_rate, 0.0, 1.0)
        _check_number("crossover_rate", self.crossover_rate, 0.0, 1.0)

        # Análisis estadístico
        _check_int("n_replicas", self.n_replicas, MIN_REPLICAS)


def read_config(config_path: Path | None) -> ExperimentConfig:
    """Lee la configuración del experimento desde un archivo TOML.

    Un `config_path` nulo (no se pasó ``--config``) produce la configuración
    por defecto. Las claves del TOML deben coincidir con los campos de
    :class:`ExperimentConfig`; cualquier otra clave es un error.
    """
    if config_path is None:
        return ExperimentConfig()

    try:
        with config_path.open(mode="rb") as f:
            data = tomllib.load(f)
    except OSError as err:
        msg = f"Could not read the config file {config_path}: {err}"
        raise ConfigError(msg) from err
    except tomllib.TOMLDecodeError as err:
        msg = f"{config_path} is not valid TOML: {err}"
        raise ConfigError(msg) from err

    known = {f.name for f in fields(ExperimentConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        msg = f"Unknown configuration keys in {config_path}: {', '.join(unknown)}"
        raise ConfigError(msg)

    return ExperimentConfig(**data)
