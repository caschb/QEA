"""Shared rover fitness evaluation and organization constraints."""

import numpy as np

from .chromosome import Chromosome


class ChromosomeEvaluator:
    """
    Puente entre los algoritmos (QEA / GA) y la clase Chromosome original.

    Responsabilidades:
      1. Guardar la MCC y las restricciones de organización definidas
         una sola vez para toda la sesión.
      2. Evaluar los genes con los mismos costos y bits protegidos que
         Chromosome.get_total_cost(), sin reconstruir el objeto por candidato.
      3. Exponer qué índices son "protegidos" (golden) para que el QEA
         no rote esos qubits y el GA no los mute.

    De esta forma QEA y GA resuelven exactamente el mismo problema.
    """

    def __init__(
        self,
        n_agents: int,
        cost_matrix: np.ndarray,
        constraints: list | None = None,
    ) -> None:
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
        row, col = np.triu_indices(n_agents, k=1)
        self.gene_costs = np.asarray(cost_matrix, dtype=float)[row, col]
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
                chrom.get_genes()[i] = 1  # protegido: siempre 1
            else:
                chrom.get_genes()[i] = int(genes[i])

        return chrom

    # ── interfaz pública ───────────────────────────────────────────────

    def evaluate(self, binary: np.ndarray) -> float:
        """
        Evalúa un cromosoma binario con la fórmula de get_total_cost().

        Esta es LA ÚNICA función de aptitud usada por QEA y GA.
        Corresponde a la Ecuación 1 de Carvajal-Godínez [6]:
            CI = Σ_i Σ_j  c_ij * A_ij(x*)
        """
        # Chromosome.get_total_cost() sums the upper triangle of the adjacency
        # matrix. Its gene order is exactly np.triu_indices(..., k=1).
        genes = np.asarray(binary)
        return float(np.dot(self.gene_costs, np.maximum(genes, self.golden_mask)))

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

    def _edge_index(self, i: int, j: int) -> int:
        """Gen de la arista (i, j), 0-indexada, en el orden de np.triu_indices."""
        i, j = min(i, j), max(i, j)
        return i * (2 * self.n_agents - i - 1) // 2 + (j - i - 1)

    def get_team_gene_groups(self) -> list[tuple[int, list[int]]]:
        """Genes que el QEA entrelaza, uno por equipo de ``constraints``.

        Cada equipo [lider, m1, m2, ...] (nodos 1-indexados) produce
        (gen líder-m1, [gen líder-m2, gen líder-m3, ...]): el primer enlace
        del equipo es el ancla que se entrelaza con los demás. Un equipo con
        un solo miembro no tiene con qué correlacionarse y no aparece.
        """
        groups = []
        for team in self.constraints:
            leader, *members = (node - 1 for node in team)
            anchor, *rest = (self._edge_index(leader, m) for m in members)
            if rest:
                groups.append((anchor, rest))
        return groups
