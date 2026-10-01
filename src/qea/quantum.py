"""Quantum chromosome rotations and Qiskit observation."""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator


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


class QiskitObserver:
    """Observación del cromosoma Q-bit mediante circuito Qiskit/Aer.

    `method` es el método de simulación de AerSimulator ("automatic",
    "statevector", "matrix_product_state", ...); ExperimentConfig lo valida
    contra los métodos que admite la versión de qiskit-aer instalada.

    Entre la preparación RY y la medición se entrelaza el gen ancla de cada
    equipo con los de sus demás miembros (``team_groups``). Se usan RXX/RZZ
    y no CX porque son simétricas y paramétricas: el líder influye en el
    miembro con una correlación graduada por theta, sin el control
    determinista de un CNOT. RZZ por sí sola solo cambia fases relativas y no
    altera las probabilidades de medición del circuito actual.

    Las compuertas conectan genes lejanos en el registro (p. ej. aristas 1-2
    y 1-8), un entrelazamiento de largo alcance que resta a
    "matrix_product_state" su ventaja sobre "statevector".
    """

    def __init__(
        self,
        method: str = "automatic",
        team_groups: list[tuple[int, list[int]]] | None = None,
        entanglement_strength: float = 0.15 * np.pi,
        entanglement_gate: str = "RXX",
    ) -> None:
        self.simulator = AerSimulator(method=method)
        self.team_groups = team_groups or []
        self.entanglement_strength = entanglement_strength
        self.entanglement_gate = entanglement_gate

    def observe(self, chromosome: QuantumChromosome) -> np.ndarray:
        n = chromosome.n_genes
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(n, "c")
        qc = QuantumCircuit(qr, cr)
        for i, (alpha, _) in enumerate(chromosome.amplitudes):
            angle = 2.0 * np.arccos(np.clip(alpha, -1.0, 1.0))
            qc.ry(angle, qr[i])
        theta = self.entanglement_strength
        for anchor, members in self.team_groups:
            for member in members:
                if self.entanglement_gate in ("RXX", "BOTH"):
                    qc.rxx(theta, qr[anchor], qr[member])
                if self.entanglement_gate in ("RZZ", "BOTH"):
                    qc.rzz(theta, qr[anchor], qr[member])
        qc.measure(qr, cr)
        counts = self.simulator.run(qc, shots=1).result().get_counts(qc)
        measured = max(counts, key=counts.get)
        return np.array([int(b) for b in reversed(measured)], dtype=int)
