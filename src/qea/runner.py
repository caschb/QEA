"""CLI orchestration for selected evolutionary algorithms."""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .analysis import run_statistical_analysis
from .config import read_config
from .evaluator import ChromosomeEvaluator
from .execution import run_algorithms
from .plotting import plot_comparison


def create_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="QEA",
        description="A quantum evolutionary algorithm",
    )
    parser.add_argument("-c", "--config", type=Path, required=False)
    return parser.parse_args()


def generate_mcc(seed: int, n_agents: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    cost_matrix = rng.uniform(1.0, 6.0, (n_agents, n_agents))
    sup_triangular = cost_matrix - np.tril(cost_matrix)
    return sup_triangular + np.diag(np.diag(cost_matrix)) + sup_triangular.T


def main() -> None:
    args = create_argument_parser()
    config = read_config(args.config)
    mcc = generate_mcc(config.seed, config.n_agents)

    evaluator = ChromosomeEvaluator(config.n_agents, mcc, config.constraints)

    print(f"\n  Genes totales   : {evaluator.n_genes}")
    print(f"  Genes protegidos: {len(evaluator.get_protected_indices())}")
    print(f"  Genes libres    : {len(evaluator.get_free_indices())}")
    print(f"  Equipos entrelazados: {len(evaluator.get_team_gene_groups())}")

    results = run_algorithms(config, evaluator)
    print("\nGenerando figura comparativa...")
    plot_comparison(results, config, evaluator)
    print(f"\nAnálisis estadístico ({config.n_replicas} réplicas)...")
    analysis = run_statistical_analysis(config, mcc)

    print("\n" + "█" * 62)
    print("  RESUMEN FINAL")
    print("█" * 62)
    for name, result in results.items():
        print(
            f"  {name}: mejor CI = {result.best_fitness:.4f} | "
            f"{result.time_elapsed:.2f}s",
        )
    best = min(result.best_fitness for result in results.values())
    winners = [name for name, result in results.items() if result.best_fitness == best]
    print(f"  Mejor resultado: {', '.join(winners)}")
    print("  Función aptitud: Chromosome.get_total_cost()")
    for pair in analysis["comparisons"]:
        print(
            f"  {pair['left']} vs {pair['right']}: "
            f"p ajustado = {pair['adjusted_p_value']:.4f} "
            f"({'significativo' if pair['significant'] else 'no significativo'})",
        )
    print("█" * 62)
