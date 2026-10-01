"""QEA testbed public API and command-line entry point."""

from .algorithms.ga import GAConfig, GeneticAlgorithm
from .algorithms.qea import QEA, QEAConfig
from .analysis import run_statistical_analysis
from .config import ExperimentConfig, read_config
from .evaluator import ChromosomeEvaluator
from .execution import run_algorithms
from .interface import EvolutionaryAlgorithm, RunContext
from .plotting import plot_comparison
from .registry import create_algorithm
from .runner import create_argument_parser, generate_mcc, main

__all__ = [
    "QEA",
    "ChromosomeEvaluator",
    "EvolutionaryAlgorithm",
    "ExperimentConfig",
    "GAConfig",
    "GeneticAlgorithm",
    "QEAConfig",
    "RunContext",
    "create_algorithm",
    "create_argument_parser",
    "generate_mcc",
    "main",
    "plot_comparison",
    "read_config",
    "run_algorithms",
    "run_statistical_analysis",
]
