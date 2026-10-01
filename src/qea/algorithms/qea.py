"""Quantum evolutionary algorithm for rover topologies."""

import time

import numpy as np

from qea.config import ExperimentConfig
from qea.evaluator import ChromosomeEvaluator
from qea.quantum import QiskitObserver, QuantumChromosome
from qea.result import AlgorithmResult


class QEA:
    """
    QEA que usa ChromosomeEvaluator como función de aptitud.

    Diferencias respecto a la versión anterior:
      - evaluate()  llama a ChromosomeEvaluator.evaluate()
      - enforce_golden() se aplica tras cada observación
      - Los qubits en golden_mask están bloqueados en el QuantumChromosome
    """

    def __init__(self, cfg: ExperimentConfig, evaluator: ChromosomeEvaluator):
        self.cfg = cfg
        self.evaluator = evaluator
        self.n_genes = evaluator.n_genes
        self.rng = np.random.default_rng(cfg.seed)
        if cfg.use_qiskit:
            self.observer = QiskitObserver(
                cfg.aer_method,
                team_groups=(
                    evaluator.get_team_gene_groups() if cfg.enable_entanglement else []
                ),
                entanglement_strength=cfg.entanglement_strength,
                entanglement_gate=cfg.entanglement_gate,
            )

    def _observe(self, chromosome: QuantumChromosome) -> np.ndarray:
        if self.cfg.use_qiskit:
            raw = self.observer.observe(chromosome)
        else:
            r = self.rng.random(self.n_genes)
            raw = (r >= (chromosome.amplitudes[:, 0] ** 2)).astype(int)
        # Forzar golden_genes DESPUÉS del colapso cuántico
        return self.evaluator.enforce_golden(raw)

    def _gdaa(self, generation: int) -> float:
        theta = self.cfg.theta_initial * np.exp(-self.cfg.decay_rate * generation)
        return max(theta, self.cfg.theta_min)

    def _diversity(self, chrom: QuantumChromosome) -> float:
        p = np.clip(chrom.prob_one, 1e-10, 1 - 1e-10)
        return float(np.mean(-p * np.log2(p) - (1 - p) * np.log2(1 - p)))

    def run(self, verbose: bool = True) -> AlgorithmResult:
        cfg = self.cfg
        locked = self.evaluator.get_protected_indices()

        # Inicialización
        chrom = QuantumChromosome(self.n_genes, locked_indices=locked)
        best_binary = self._observe(chrom)
        best_fitness = self.evaluator.evaluate(best_binary)

        f_hist, b_hist, d_hist = [], [], []

        if verbose:
            print(f"\n{'=' * 62}")
            print(f"  QEA (integrado con Chromosome) — {cfg.n_agents} agentes")
            print(
                f"  Genes libres: {len(self.evaluator.get_free_indices())} / {self.n_genes}  "
                f"| Genes protegidos: {len(locked)}",
            )
            print(
                f"  Escenario: {cfg.scenario.upper()} | θ₀={cfg.theta_initial / np.pi:.4f}π",
            )
            entanglement = (
                f"ON ({cfg.entanglement_gate})" if cfg.enable_entanglement else "OFF"
            )
            n_teams = len(self.evaluator.get_team_gene_groups())
            print(
                f"  Entrelazamiento: {entanglement}"
                f" | θ_entrelazamiento={cfg.entanglement_strength / np.pi:.4f}π"
                f" | equipos entrelazados: {n_teams}",
            )
            print(f"{'=' * 62}")
            print(
                f"{'Gen':>6} {'CI actual':>12} {'CI mejor':>12} {'θ':>10} {'Diversidad':>11}",
            )
            print(f"{'-' * 62}")

        t0 = time.time()
        for gen in range(cfg.max_generations):
            theta = self._gdaa(gen)
            obs = self._observe(chrom)
            curr_f = self.evaluator.evaluate(obs)

            if curr_f < best_fitness:
                best_fitness = curr_f
                best_binary = obs.copy()

            chrom.apply_rotation(
                obs, best_binary, curr_f, best_fitness, theta, cfg.rotation_scheme,
            )

            div = self._diversity(chrom)
            f_hist.append(curr_f)
            b_hist.append(best_fitness)
            d_hist.append(div)

            if verbose and (gen % 25 == 0 or gen == cfg.max_generations - 1):
                print(
                    f"{gen:>6} {curr_f:>12.3f} {best_fitness:>12.3f} "
                    f"{theta / np.pi:>9.5f}π {div:>11.4f}",
                )

        elapsed = time.time() - t0
        if verbose:
            print(
                f"\n  ✓ QEA completado en {elapsed:.2f}s | Mejor CI = {best_fitness:.4f}",
            )

        return AlgorithmResult(
            best_binary=best_binary,
            best_fitness=best_fitness,
            fitness_history=f_hist,
            best_history=b_hist,
            diversity_history=d_hist,
            time_elapsed=elapsed,
            label="QEA",
        )
