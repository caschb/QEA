"""The contract between experiment execution and evolutionary algorithms."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .evaluator import ChromosomeEvaluator
from .result import AlgorithmResult
from .validation import SCENARIOS, _check_bool, _check_choice, _check_int


@dataclass(frozen=True)
class RunContext:
    """Shared inputs for one independent algorithm run."""

    evaluator: ChromosomeEvaluator
    seed: int = 42
    max_generations: int = 150
    scenario: str = "nominal"
    verbose: bool = True

    def __post_init__(self) -> None:
        _check_bool("verbose", self.verbose)
        _check_int("seed", self.seed, 0)
        _check_int("max_generations", self.max_generations, 1)
        _check_choice("scenario", self.scenario, SCENARIOS)


@runtime_checkable
class EvolutionaryAlgorithm(Protocol):
    """Algorithms own their parameters and return the standard result."""

    def run(self, context: RunContext) -> AlgorithmResult: ...
