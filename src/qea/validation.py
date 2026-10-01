"""Shared configuration validation helpers."""

from functools import cache

from qiskit_aer import AerSimulator


class ConfigError(Exception): ...


SCENARIOS = ("nominal", "safe", "critical")
ROTATION_SCHEMES = ("I", "II", "III")
ENTANGLEMENT_GATES = ("RXX", "RZZ", "BOTH")

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
