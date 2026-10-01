"""Execute selected algorithms through their common interface."""

from .config import ExperimentConfig
from .evaluator import ChromosomeEvaluator
from .interface import RunContext
from .registry import create_algorithm
from .result import AlgorithmResult


def run_algorithms(
    cfg: ExperimentConfig,
    evaluator: ChromosomeEvaluator,
    *,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, AlgorithmResult]:
    context = RunContext(
        evaluator,
        cfg.seed if seed is None else seed,
        cfg.max_generations,
        cfg.scenario,
        verbose,
    )
    results = {}
    for name in cfg.selected_algorithms:
        if verbose:
            print(f"\nEjecutando {name}...")
        results[name] = create_algorithm(name, cfg.algorithms[name]).run(context)
    return results
