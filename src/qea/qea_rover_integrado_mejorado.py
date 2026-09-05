"""
=============================================================================
qea_rover_integrado_mejorado.py
=============================================================================
QEA vs GA — Comparación justa usando la clase Chromosome del
Dr. Johan Carvajal Godínez como evaluador compartido.

CAMBIO CLAVE respecto a la versión anterior:
  - Ambos algoritmos usan ChromosomeEvaluator como única fuente de verdad
  - Las mismas restricciones de organización (team/hierarchy) se aplican
    a QEA y GA antes de evaluar → espacio de búsqueda idéntico
  - La función de aptitud es get_total_cost() de la clase Chromosome
    (Ecuación 1 de Carvajal-Godínez [6]) en ambos casos
  - Los genes protegidos (golden_genes) son respetados en QEA mediante
    "qubits anclados" que no reciben rotación

Beca CeNAT-CONARE | ITCR — CNCA
Autor: Jonathan D. Fuentes Calvo
Tutor: Dr. Ing. Johan Carvajal Godínez
=============================================================================
"""

import time
from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from scipy import stats

# Clase original del Dr. Carvajal — NO se modifica
from .chromosome import Chromosome
from .config import ExperimentConfig

# =============================================================================
# 1. EVALUADOR COMPARTIDO — el núcleo de la comparación justa
# =============================================================================


class ChromosomeEvaluator:
    """
    Puente entre los algoritmos (QEA / GA) y la clase Chromosome original.

    Responsabilidades:
      1. Guardar la MCC y las restricciones de organización definidas
         una sola vez para toda la sesión.
      2. Dado un arreglo binario (genes), inyectarlo en un Chromosome,
         forzar los golden_genes y retornar get_total_cost().
      3. Exponer qué índices son "protegidos" (golden) para que el QEA
         no rote esos qubits y el GA no los mute.

    De esta forma QEA y GA resuelven exactamente el mismo problema.
    """

    def __init__(
        self, n_agents: int, cost_matrix: np.ndarray, constraints: list | None = None,
    ):
        """
        Parámetros
        ----------
        n_agents    : número de subsistemas del rover
        cost_matrix : MCC  shape (n_agents, n_agents), float
        constraints : lista de listas. Cada sub-lista es una restricción.
                      El primer elemento es el nodo maestro.
                      Ejemplo: [[1,2,3], [4,5]]  →  dos equipos
                      Pasar None para no imponer restricciones.
        """
        self.n_agents = n_agents
        self.n_genes = n_agents * (n_agents - 1) // 2
        self.cost_matrix = cost_matrix.tolist()  # Chromosome espera lista Python
        self.constraints = constraints or []

        # Construir un Chromosome de referencia solo para extraer
        # el patrón de golden_genes (bits protegidos)
        ref = self._build_chromosome([0] * self.n_genes)
        self.golden_mask = np.array(ref.get_golden_genes(), dtype=int)
        # golden_mask[i] == 1  →  gen i está protegido (siempre vale 1)

    # ── helpers internos ───────────────────────────────────────────────

    def _build_chromosome(self, genes: list) -> Chromosome:
        """Crea un Chromosome, aplica restricciones y sobreescribe genes."""
        chrom = Chromosome(self.n_agents, self.cost_matrix)

        # Aplicar las restricciones de organización
        for c in self.constraints:
            chrom.set_constraint_org_team(c)

        # Sobreescribir genes con el arreglo recibido,
        # pero RESPETAR los golden_genes (bits forzados a 1)
        golden = chrom.get_golden_genes()
        for i in range(self.n_genes):
            if golden[i] == 1:
                chrom._genes[i] = 1  # protegido: siempre 1
            else:
                chrom._genes[i] = int(genes[i])

        return chrom

    # ── interfaz pública ───────────────────────────────────────────────

    def evaluate(self, binary: np.ndarray) -> float:
        """
        Evalúa un cromosoma binario y retorna get_total_cost().

        Esta es LA ÚNICA función de aptitud usada por QEA y GA.
        Corresponde a la Ecuación 1 de Carvajal-Godínez [6]:
            CI = Σ_i Σ_j  c_ij * A_ij(x*)
        """
        chrom = self._build_chromosome(binary.tolist())
        return chrom.get_total_cost()

    def enforce_golden(self, binary: np.ndarray) -> np.ndarray:
        """
        Fuerza a 1 los bits protegidos en un arreglo binario.
        Se llama después de cada observación/mutación para mantener
        la coherencia con las restricciones de organización.
        """
        result = binary.copy()
        result[self.golden_mask == 1] = 1
        return result

    def get_chromosome_object(self, binary: np.ndarray) -> Chromosome:
        """Retorna el objeto Chromosome completo para análisis / visualización."""
        return self._build_chromosome(binary.tolist())

    def get_protected_indices(self) -> np.ndarray:
        """Índices de genes que no deben ser mutados/rotados."""
        return np.where(self.golden_mask == 1)[0]

    def get_free_indices(self) -> np.ndarray:
        """Índices de genes sobre los que operan los algoritmos."""
        return np.where(self.golden_mask == 0)[0]


# =============================================================================
# 2. RESULTADO ESTÁNDAR
# =============================================================================


@dataclass
class AlgorithmResult:
    """Resultado estándar, igual para QEA y GA."""

    best_binary: np.ndarray
    best_fitness: float
    fitness_history: list[float] = field(default_factory=list)
    best_history: list[float] = field(default_factory=list)
    diversity_history: list[float] = field(default_factory=list)
    time_elapsed: float = 0.0
    label: str = ""


# =============================================================================
# 3. CROMOSOMA CUÁNTICO Q-BIT
# =============================================================================


class QuantumChromosome:
    """
    Cromosoma cuántico con codificación Q-bit.
    Inicialización: α = β = 1/√2  (superposición uniforme).

    Los qubits en 'locked_indices' se fijan en β=1 (P(1)=1)
    para reflejar los golden_genes de Chromosome.
    """

    def __init__(self, n_genes: int, locked_indices: np.ndarray | None = None) -> None:
        self.n_genes = n_genes
        self.amplitudes = np.full((n_genes, 2), 1.0 / np.sqrt(2))
        self.locked = (
            locked_indices if locked_indices is not None else np.array([], dtype=int)
        )
        # Fijar qubits protegidos: α=0, β=1 → P(1)=1 siempre
        for idx in self.locked:
            self.amplitudes[idx] = [0.0, 1.0]

    @property
    def prob_one(self):
        return self.amplitudes[:, 1] ** 2

    def apply_rotation(
        self,
        observed: np.ndarray,
        best: np.ndarray,
        curr_fit: float,
        best_fit: float,
        theta: float,
        scheme: str = "I",
    ) -> None:
        """
        Actualización del cromosoma cuántico mediante Puerta de Rotación (QRG).
        Ref: Xiong et al. (2018) [13]

        Tres esquemas disponibles — se selecciona con el parámetro 'scheme':

        Scheme I  : 8 condiciones. Combina (xi, xb, f_worse) para decidir
                    dirección. También aplica rotación suave (θ×0.5) cuando
                    los bits difieren aunque la solución no sea peor.
                    → Más expresivo, mejor exploración, más complejo.

        Scheme II : 1 condición. Rota SOLO si los bits difieren Y la solución
                    actual es peor que la mejor conocida.
                    → Conservador: no toca qubits cuando la solución es buena.

        Scheme III: 1 condición. Rota siempre que los bits difieren,
                    sin importar si la solución es mejor o peor.
                    → Agresivo: máxima presión hacia la élite en todo momento.

        Comparación resumida:
          Condición para rotar       Scheme I   Scheme II   Scheme III
          xi≠xb  AND  f_worse           ✓           ✓           ✓
          xi≠xb  AND  NOT f_worse       ✓ (×0.5)    ✗           ✓
        """
        f_worse = curr_fit > best_fit  # True si la solución actual es peor

        for i in range(self.n_genes):
            if i in self.locked:  # qubit protegido: no rotar nunca
                continue

            xi, xb = int(observed[i]), int(best[i])
            alpha, beta = self.amplitudes[i]
            delta = 0.0

            # ── Scheme I ──────────────────────────────────────────────
            # 8 condiciones: considera xi, xb, f_worse y signo de α×β
            if scheme == "I":
                if xi == 1 and xb == 0 and f_worse:
                    # Colapsó a 1 pero élite tiene 0 → empujar hacia 0
                    delta = -theta if (alpha * beta > 0) else theta
                elif xi == 0 and xb == 1 and f_worse:
                    # Colapsó a 0 pero élite tiene 1 → empujar hacia 1
                    delta = theta if (alpha * beta > 0) else -theta
                elif xi != xb:
                    # Bits difieren pero solución no es peor → rotación suave
                    delta = (1 if xb == 1 else -1) * theta * 0.5

            # ── Scheme II ─────────────────────────────────────────────
            # Rota SOLO si los bits difieren Y la solución es peor.
            # Si la solución actual es buena, no toca los qubits.
            # Más conservador que I y III → converge más lento pero
            # preserva mejor las soluciones que ya funcionan bien.
            elif scheme == "II":
                if xi != xb and f_worse:
                    delta = theta if xb == 1 else -theta

            # ── Scheme III ────────────────────────────────────────────
            # Rota siempre que los bits difieren, sin considerar fitness.
            # Presión constante hacia la élite → convergencia rápida
            # pero mayor riesgo de pérdida de diversidad prematura.
            elif scheme == "III":
                if xi != xb:
                    delta = theta if xb == 1 else -theta

            # ── Aplicar rotación y renormalizar ───────────────────────
            if abs(delta) > 1e-10:
                c, s = np.cos(delta), np.sin(delta)
                na, nb = c * alpha - s * beta, s * alpha + c * beta
                norm = np.sqrt(na**2 + nb**2)
                self.amplitudes[i] = [na / norm, nb / norm]


# =============================================================================
# 4. OBSERVADOR QISKIT
# =============================================================================


class QiskitObserver:
    """Observación del cromosoma Q-bit mediante circuito Qiskit/Aer.

    `method` es el método de simulación de AerSimulator ("automatic",
    "statevector", "matrix_product_state", ...); ExperimentConfig lo valida
    contra los métodos que admite la versión de qiskit-aer instalada.
    """

    def __init__(self, method: str = "automatic") -> None:
        self.simulator = AerSimulator(method=method)

    def observe(self, chromosome: QuantumChromosome) -> np.ndarray:
        n = chromosome.n_genes
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(n, "c")
        qc = QuantumCircuit(qr, cr)
        for i, (alpha, _) in enumerate(chromosome.amplitudes):
            angle = 2.0 * np.arccos(np.clip(alpha, -1.0, 1.0))
            qc.ry(angle, qr[i])
        qc.measure(qr, cr)
        counts = self.simulator.run(qc, shots=1).result().get_counts(qc)
        measured = max(counts, key=counts.get)
        return np.array([int(b) for b in reversed(measured)], dtype=int)


# =============================================================================
# 5. QEA INTEGRADO
# =============================================================================


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
            self.observer = QiskitObserver(cfg.aer_method)

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


# =============================================================================
# 6. GA INTEGRADO  (usa exactamente la misma ChromosomeEvaluator)
# =============================================================================


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
        fits = np.array([self.evaluator.evaluate(ind) for ind in pop])
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
            fits = np.array([self.evaluator.evaluate(ind) for ind in pop])

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


# =============================================================================
# 7. ANÁLISIS ESTADÍSTICO (Objetivo Específico 2)
# =============================================================================


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
            use_qiskit=False,  # rápido para réplicas masivas
            seed=cfg.seed + rep * 137,
        )
        # Misma evaluador para ambos en esta réplica
        ev = ChromosomeEvaluator(rep_cfg.n_agents, cost_matrix, rep_cfg.constraints)

        qea_r = QEA(rep_cfg, ev).run(verbose=False)
        ga_r = GeneticAlgorithm(rep_cfg, ev).run(verbose=False)

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


# =============================================================================
# 8. VISUALIZACIÓN
# =============================================================================


def plot_comparison(
    qea_r: AlgorithmResult,
    ga_r: AlgorithmResult,
    cfg: ExperimentConfig,
    evaluator: ChromosomeEvaluator,
    save_path: str = "resultados_integrado.png",
):

    fig = plt.figure(figsize=(16, 10), facecolor="#0a0f18")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    bg, grid_c, txt = "#0d1b2a", "#1e3a4a", "#c8d8e8"

    def sax(ax, title):
        ax.set_facecolor(bg)
        ax.tick_params(colors=txt, labelsize=7)
        for s in ax.spines.values():
            s.set_color(grid_c)
        ax.grid(True, color=grid_c, lw=0.4, alpha=0.7)
        ax.set_title(title, color=txt, fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.label.set_color(txt)
        ax.yaxis.label.set_color(txt)

    gens = range(cfg.max_generations)

    # ── Convergencia ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(
        gens,
        qea_r.best_history,
        color="#00d4aa",
        lw=2.5,
        label="QEA — Mejor CI (get_total_cost)",
        zorder=3,
    )
    ax1.plot(
        gens,
        qea_r.fitness_history,
        color="#00d4aa",
        lw=1,
        alpha=0.3,
        ls="--",
        label="QEA — CI actual",
    )
    ax1.plot(
        gens,
        ga_r.best_history,
        color="#4fc3f7",
        lw=2.5,
        label="GA  — Mejor CI (get_total_cost)",
        zorder=3,
    )
    ax1.plot(
        gens,
        ga_r.fitness_history,
        color="#4fc3f7",
        lw=1,
        alpha=0.3,
        ls="--",
        label="GA  — CI media",
    )
    ax1.legend(
        fontsize=7.5, facecolor=bg, labelcolor=txt, framealpha=0.85, edgecolor=grid_c,
    )
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Costo CI  [get_total_cost()]")
    sax(
        ax1,
        f"Convergencia — {cfg.n_agents} agentes | {cfg.scenario.upper()} "
        f"| Genes protegidos: {len(evaluator.get_protected_indices())}",
    )

    # ── Diversidad ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(gens, qea_r.diversity_history, color="#00d4aa", lw=1.5, label="QEA")
    ax2.plot(gens, ga_r.diversity_history, color="#4fc3f7", lw=1.5, label="GA")
    ax2.legend(fontsize=7, facecolor=bg, labelcolor=txt, edgecolor=grid_c)
    ax2.set_xlabel("Generación")
    ax2.set_ylabel("Diversidad")
    sax(ax2, "Diversidad Poblacional")

    # ── Topologías ──────────────────────────────────────────────────────
    # Nombres automáticos: escala a cualquier n_agents sin cambiar nada
    agent_names = [f"A{i + 1}" for i in range(cfg.n_agents)]

    def draw_topo(ax, binary, node_color, title):
        ax.set_facecolor(bg)
        ax.axis("off")
        ax.set_title(title, color=txt, fontsize=8.5, fontweight="bold")
        chrom = evaluator.get_chromosome_object(binary)
        golden = chrom.get_golden_genes()
        r = 1.0
        angs = [2 * np.pi * i / cfg.n_agents - np.pi / 2 for i in range(cfg.n_agents)]
        pos = [(r * np.cos(a), r * np.sin(a)) for a in angs]
        idx = 0
        mx = np.max(evaluator.cost_matrix) or 1.0
        for i in range(cfg.n_agents - 1):
            for j in range(i + 1, cfg.n_agents):
                if binary[idx] == 1:
                    t = evaluator.cost_matrix[i][j] / mx
                    # Enlace protegido → línea sólida dorada
                    # Enlace libre    → color según costo
                    lc = (
                        (0.9, 0.7, 0.0, 0.9)
                        if golden[idx] == 1
                        else (t, 1 - t * 0.6, 0.3, 0.7)
                    )
                    lw = 2.0 if golden[idx] == 1 else 0.8 + t * 2.5
                    ax.plot(
                        [pos[i][0], pos[j][0]],
                        [pos[i][1], pos[j][1]],
                        color=lc,
                        lw=lw,
                        zorder=1,
                    )
                idx += 1
        for i, (x, y) in enumerate(pos):
            ax.add_patch(
                plt.Circle((x, y), 0.13, color="#0d1b2a", ec=node_color, lw=2, zorder=2),
            )
            ax.text(
                x,
                y,
                agent_names[i],
                ha="center",
                va="center",
                fontsize=7,
                color=node_color,
                fontweight="bold",
                zorder=3,
                fontfamily="monospace",
            )
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.text(
            0,
            -1.32,
            "── protegido (golden_gene=1)  ── libre",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#888",
            fontfamily="monospace",
        )

    ax3 = fig.add_subplot(gs[1, 0])
    draw_topo(
        ax3,
        qea_r.best_binary,
        "#00d4aa",
        f"Topología QEA\nCI = {qea_r.best_fitness:.3f}",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    draw_topo(
        ax4,
        ga_r.best_binary,
        "#4fc3f7",
        f"Topología GA\nCI = {ga_r.best_fitness:.3f}",
    )

    # ── Métricas ────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(bg)
    ax5.axis("off")
    ax5.set_title(
        "Métricas Comparativas", color=txt, fontsize=9, fontweight="bold", pad=7,
    )
    winner = "QEA ✓" if qea_r.best_fitness < ga_r.best_fitness else "GA ✓"
    win_c = "#00d4aa" if "QEA" in winner else "#4fc3f7"
    qea_imp = (
        (qea_r.fitness_history[0] - qea_r.best_fitness)
        / max(qea_r.fitness_history[0], 1e-9)
        * 100
    )
    ga_imp = (
        (ga_r.fitness_history[0] - ga_r.best_fitness)
        / max(ga_r.fitness_history[0], 1e-9)
        * 100
    )
    rows = [
        ("Métrica", "QEA", "GA"),
        ("Mejor CI", f"{qea_r.best_fitness:.3f}", f"{ga_r.best_fitness:.3f}"),
        ("Mejora %", f"{qea_imp:.1f}%", f"{ga_imp:.1f}%"),
        ("Tiempo (s)", f"{qea_r.time_elapsed:.2f}", f"{ga_r.time_elapsed:.2f}"),
        ("Fn. aptitud", "get_total_cost()", "get_total_cost()"),
        ("Ganador", winner, ""),
    ]
    y = 0.93
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = (
                "#ffd700"
                if ri == 0
                else (win_c if ri == len(rows) - 1 and ci == 1 else txt)
            )
            ax5.text(
                0.03 + ci * 0.34,
                y,
                val,
                transform=ax5.transAxes,
                fontsize=7.5,
                color=c,
                fontweight="bold" if ri == 0 else "normal",
                fontfamily="monospace",
            )
        y -= 0.14
        if ri == 0:
            line = plt.Line2D(
                [0.02, 0.98],
                [y + 0.07, y + 0.07],
                transform=ax5.transAxes,
                color=grid_c,
                lw=0.8,
            )
            ax5.add_line(line)

    protected_n = len(evaluator.get_protected_indices())
    fig.suptitle(
        f"QEA vs GA — Evaluación compartida: Chromosome.get_total_cost()\n"
        f"n={cfg.n_agents} agentes | {evaluator.n_genes} genes "
        f"({protected_n} protegidos, {evaluator.n_genes - protected_n} libres) | "
        f"Escenario: {cfg.scenario}",
        color=txt,
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0f18")
    print(f"  ✓ Figura guardada: {save_path}")
