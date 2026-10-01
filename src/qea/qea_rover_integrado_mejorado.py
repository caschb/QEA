"""Compatibility imports for the original integrated experiment module.

Implementations now live in focused modules; existing imports remain supported.
"""

from .algorithms.ga import GeneticAlgorithm
from .algorithms.qea import QEA
from .analysis import run_statistical_analysis
from .evaluator import ChromosomeEvaluator
from .plotting import plot_comparison
from .quantum import QiskitObserver, QuantumChromosome
from .result import AlgorithmResult

__all__ = [
    "QEA",
    "AlgorithmResult",
    "ChromosomeEvaluator",
    "GeneticAlgorithm",
    "QiskitObserver",
    "QuantumChromosome",
    "plot_comparison",
    "run_statistical_analysis",
]
