"""Genetic algorithm for rover topologies."""

import time

import numpy as np

from qea.config import ExperimentConfig
from qea.evaluator import ChromosomeEvaluator
from qea.result import AlgorithmResult


class GeneticAlgorithm:
    """
    GA original del Dr. Carvajal adaptado para usar ChromosomeEvaluator.

    Cambios respecto a la implementación previa:
      - fitness → evaluator.evaluate()   (misma Ec. 1 que QEA)
      - enforce_golden() se aplica tras mutación → bits protegidos no mutan
      - El operador de mutación solo actúa sobre get_free_indices()
    """

    def __init__(self, cfg: ExperimentConfig, evaluator: ChromosomeEvaluator):
        self.cfg = cfg
        self.evaluator = evaluator
        self.n_genes = evaluator.n_genes
        self.rng = np.random.default_rng(cfg.seed + 1000)
        self.free_idx = evaluator.get_free_indices()  # solo estos genes mutan

    def _init_population(self) -> np.ndarray:
        pop = self.rng.integers(0, 2, size=(self.cfg.pop_size, self.n_genes))
        # Forzar golden_genes en toda la población inicial
        for i in range(self.cfg.pop_size):
            pop[i] = self.evaluator.enforce_golden(pop[i])
        return pop

    def _tournament(self, pop: np.ndarray, fits: np.ndarray, k: int = 3) -> np.ndarray:
        idx = self.rng.choice(len(pop), k, replace=False)
        return pop[idx[np.argmin(fits[idx])]].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray):
        if self.rng.random() < self.cfg.crossover_rate:
            pt = self.rng.integers(1, self.n_genes)
            c1 = np.concatenate([p1[:pt], p2[pt:]])
            c2 = np.concatenate([p2[:pt], p1[pt:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        return c1, c2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        """Mutación bit-flip solo sobre genes LIBRES (no golden)."""
        result = ind.copy()
        for i in self.free_idx:
            if self.rng.random() < self.cfg.mutation_rate:
                result[i] = 1 - result[i]
        # enforce_golden por seguridad (cruce puede haber alterado algo)
        return self.evaluator.enforce_golden(result)

    def run(self, verbose: bool = True) -> AlgorithmResult:
        cfg = self.cfg
        pop = self._init_population()
        fitness_cache: dict[bytes, float] = {}

        def fitness(ind: np.ndarray) -> float:
            key = ind.tobytes()
            if key not in fitness_cache:
                fitness_cache[key] = self.evaluator.evaluate(ind)
            return fitness_cache[key]

        fits = np.array([fitness(ind) for ind in pop])
        best_idx = np.argmin(fits)
        best_binary = pop[best_idx].copy()
        best_fitness = fits[best_idx]

        f_hist, b_hist, d_hist = [], [], []

        if verbose:
            print(f"\n{'=' * 62}")
            print(f"  GA (integrado con Chromosome) — {cfg.n_agents} agentes")
            print(
                f"  Población: {cfg.pop_size} | Mutación: {cfg.mutation_rate} "
                f"| Cruce: {cfg.crossover_rate}",
            )
            print(
                f"  Genes libres: {len(self.free_idx)} / {self.n_genes}  "
                f"| Genes protegidos: {len(self.evaluator.get_protected_indices())}",
            )
            print(f"{'=' * 62}")

        t0 = time.time()
        for gen in range(cfg.max_generations):
            new_pop = []
            while len(new_pop) < cfg.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])

            pop = np.array(new_pop[: cfg.pop_size])
            fits = np.array([fitness(ind) for ind in pop])

            if fits.min() < best_fitness:
                best_fitness = fits.min()
                best_binary = pop[np.argmin(fits)].copy()

            f_hist.append(float(np.mean(fits)))
            b_hist.append(best_fitness)
            d_hist.append(float(np.mean(np.std(pop[:, self.free_idx], axis=0))))

            if verbose and (gen % 25 == 0 or gen == cfg.max_generations - 1):
                print(
                    f"  Gen {gen:>4} | CI media: {np.mean(fits):>10.3f} "
                    f"| CI mejor: {best_fitness:>10.3f}",
                )

        elapsed = time.time() - t0
        if verbose:
            print(
                f"\n  ✓ GA completado en {elapsed:.2f}s | Mejor CI = {best_fitness:.4f}",
            )

        return AlgorithmResult(
            best_binary=best_binary,
            best_fitness=best_fitness,
            fitness_history=f_hist,
            best_history=b_hist,
            diversity_history=d_hist,
            time_elapsed=elapsed,
            label="GA",
        )
