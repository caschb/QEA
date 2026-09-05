import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import ExperimentConfig, read_config
from .qea_rover_integrado_mejorado import (
    QEA,
    ChromosomeEvaluator,
    GeneticAlgorithm,
    plot_comparison,
    run_statistical_analysis,
)

__all__ = [
    "QEA",
    "ChromosomeEvaluator",
    "ExperimentConfig",
    "GeneticAlgorithm",
    "generate_mcc",
    "main",
    "plot_comparison",
    "read_config",
    "run_statistical_analysis",
]

# Las réplicas del análisis estadístico se acotan a este número de
# generaciones para que decenas de corridas sigan siendo viables.
STATS_MAX_GENERATIONS = 50


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

    # ── Ejecutar QEA ───────────────────────────────────────────────────
    print("\n[1/4] Ejecutando QEA...")
    qea_result = QEA(config, evaluator).run(verbose=True)

    # ── Ejecutar GA ────────────────────────────────────────────────────
    print("\n[2/4] Ejecutando GA (Dr. Carvajal)...")
    ga_result = GeneticAlgorithm(config, evaluator).run(verbose=True)

    # ── Visualizar ─────────────────────────────────────────────────────
    print("\n[3/4] Generando figura comparativa...")
    plot_comparison(qea_result, ga_result, config, evaluator)

    # ── Análisis estadístico ───────────────────────────────────────────
    print(f"\n[4/4] Análisis estadístico ({config.n_replicas} réplicas)...")
    stats_config = replace(
        config,
        max_generations=min(config.max_generations, STATS_MAX_GENERATIONS),
        use_qiskit=False,  # siempre clásico para réplicas masivas
    )
    stats = run_statistical_analysis(stats_config, mcc, verbose=True)

    # ── Resumen final ──────────────────────────────────────────────────
    print("\n" + "█" * 62)
    print("  RESUMEN FINAL")
    print("█" * 62)
    delta = ga_result.best_fitness - qea_result.best_fitness
    print(f"  QEA mejor CI     : {qea_result.best_fitness:.4f}")
    print(f"  GA  mejor CI     : {ga_result.best_fitness:.4f}")
    print(
        f"  Δ (GA - QEA)     : {delta:+.4f}  "
        f"({'QEA mejor' if delta > 0 else 'GA mejor' if delta < 0 else 'empate'})",
    )
    print("  Función aptitud  : Chromosome.get_total_cost()  ← idéntica en ambos")
    print(
        f"  Genes protegidos : {len(evaluator.get_protected_indices())}  "
        f"← respetados por QEA y GA",
    )
    print(
        f"  Wilcoxon p-valor : {stats['p_value']:.4f}  "
        f"({'significativo' if stats['significant'] else 'no significativo'})",
    )
    print("█" * 62)
