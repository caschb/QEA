"""Repeated QEA and GA runs and statistical comparison."""

from dataclasses import replace

import numpy as np
from scipy import stats

from .config import ExperimentConfig
from .evaluator import ChromosomeEvaluator
from .interface import RunContext
from .registry import create_algorithm


def run_statistical_analysis(
    cfg: ExperimentConfig,
    cost_matrix: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    ≥30 réplicas con semillas distintas.
    Prueba de Wilcoxon (no paramétrica, α=0.05).
    Misma ChromosomeEvaluator en QEA y GA → comparación válida.
    """
    n_replicas = cfg.n_replicas
    qea_scores, ga_scores = [], []
    qea_times, ga_times = [], []

    print(f"\n{'=' * 62}")
    print(f"  ANÁLISIS ESTADÍSTICO — {n_replicas} réplicas")
    print(f"  Escenario: {cfg.scenario} | n_agents: {cfg.n_agents}")
    print(f"{'=' * 62}")

    for rep in range(n_replicas):
        # replace() conserva el resto de los parámetros de cfg (decay_rate,
        # rotation_scheme, pop_size, mutation_rate, crossover_rate): las
        # réplicas deben diferir de la corrida principal solo en la semilla.
        rep_cfg = replace(
            cfg,
            seed=cfg.seed + rep * 137,
        )
        # Misma evaluador para ambos en esta réplica
        ev = ChromosomeEvaluator(rep_cfg.n_agents, cost_matrix, rep_cfg.constraints)

        context = RunContext(
            ev, rep_cfg.seed, rep_cfg.max_generations, rep_cfg.scenario, False
        )
        qea_parameters = replace(rep_cfg.algorithms["qea"], use_qiskit=False)
        qea_r = create_algorithm("qea", qea_parameters).run(context)
        ga_r = create_algorithm("ga", rep_cfg.algorithms["ga"]).run(context)

        qea_scores.append(qea_r.best_fitness)
        ga_scores.append(ga_r.best_fitness)
        qea_times.append(qea_r.time_elapsed)
        ga_times.append(ga_r.time_elapsed)

        if verbose:
            print(
                f"  Rep {rep + 1:>3}/{n_replicas} | "
                f"QEA: {qea_r.best_fitness:>8.3f} | "
                f"GA:  {ga_r.best_fitness:>8.3f}",
            )

    qea_arr = np.array(qea_scores)
    ga_arr = np.array(ga_scores)

    # Wilcoxon: H₀ = distribuciones iguales / H₁ = QEA < GA
    stat, p_val = stats.wilcoxon(qea_arr, ga_arr, alternative="less")

    results = {
        "qea_mean": float(np.mean(qea_arr)),
        "qea_std": float(np.std(qea_arr)),
        "qea_best": float(np.min(qea_arr)),
        "ga_mean": float(np.mean(ga_arr)),
        "ga_std": float(np.std(ga_arr)),
        "ga_best": float(np.min(ga_arr)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
        "winner": "QEA" if np.mean(qea_arr) < np.mean(ga_arr) else "GA",
        "qea_scores": qea_scores,
        "ga_scores": ga_scores,
    }

    print(f"\n{'─' * 62}")
    print(
        f"  QEA : μ={results['qea_mean']:>8.3f} ± {results['qea_std']:.3f} "
        f"| mejor={results['qea_best']:.3f}",
    )
    print(
        f"  GA  : μ={results['ga_mean']:>8.3f} ± {results['ga_std']:.3f} "
        f"| mejor={results['ga_best']:.3f}",
    )
    print(
        f"  Wilcoxon : W={stat:.3f}, p={p_val:.4f}  "
        f"({'SIGNIFICATIVO ✓' if p_val < 0.05 else 'no significativo ✗'})",
    )
    print(f"  Ganador  : {results['winner']}")
    return results
