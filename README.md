# QEA vs GA — Optimización de topologías MAS para arquitecturas de rover

Comparación entre un **Algoritmo Evolutivo Cuántico (QEA)** y un **Algoritmo Genético (GA)** clásico para encontrar la topología de comunicación de menor costo en un sistema multi-agente (MAS), sujeta a restricciones de organización (equipos y jerarquías).

Ambos algoritmos comparten **exactamente la misma función de aptitud** — `Chromosome.get_total_cost()` — a través de un evaluador puente, de modo que la comparación es justa: mismo espacio de búsqueda, mismas restricciones, mismo criterio de costo.

> Beca CeNAT-CONARE | ITCR — CNCA
> Autor: Jonathan D. Fuentes Calvo · Tutor: Dr. Ing. Johan Carvajal Godínez

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `chromosome.py` | Clase `Chromosome` original (Dr. Carvajal). Codifica una topología como vector binario, aplica restricciones organizacionales y calcula el costo total. **No se modifica.** |
| `qea_rover_integrado_mejorado.py` | Script principal: evaluador compartido, QEA, GA, análisis estadístico y visualización. Punto de entrada interactivo. |

---

## Requisitos

- Python 3.13 o superior

```bash
uv sync
```

`qiskit` y `qiskit-aer` solo son necesarios si se usa el modo de observación cuántica real. El script también ofrece un modo clásico (muestreo pseudoaleatorio) que reproduce el colapso del Q-bit sin simulador.

---

## Uso rápido

```bash
git clone https://github.com/johan-carvajal-godinez/QEA.git
cd QEA
uv run qea
```

El script abre un **diálogo interactivo**. Presiona Enter en cualquier pregunta para aceptar el valor por defecto entre corchetes.

### Parámetros que se solicitan

**1. Problema**

| Parámetro | Rango | Defecto | Significado |
|---|---|---|---|
| `n_agents` | 4–30 | 8 | Subsistemas del rover. Genera `n(n-1)/2` genes |
| `scenario` | nominal / safe / critical | nominal | Etiqueta del escenario de misión |
| restricciones | — | ninguna | Equipos de organización (ver abajo) |

**2. QEA**

| Parámetro | Rango | Defecto | Significado |
|---|---|---|---|
| `rotation_scheme` | I / II / III | I | Esquema de la puerta de rotación cuántica |
| `max_generations` | 10–500 | 150 | Iteraciones |
| `θ₀` (fracción de π) | 0.001–0.1 | 0.05 | Ángulo de rotación inicial |
| `λ` (GDAA) | 0.001–0.1 | 0.02 | Tasa de decaimiento exponencial de θ |
| Qiskit | s/n | s si n≤15 | Observación con `AerSimulator` vs. modo clásico |

**3. GA**

| Parámetro | Rango | Defecto |
|---|---|---|
| `pop_size` | 10–200 | 30 |
| `mutation_rate` | 0.001–0.2 | 0.02 |
| `crossover_rate` | 0.1–1.0 | 0.8 |

**4. Estadística**

| Parámetro | Rango | Defecto |
|---|---|---|
| `n_replicas` | 2–100 | 30 |
| `seed` | 0–99999 | 42 |

### Definir restricciones de organización

Cada restricción es una lista de nodos separados por espacio. **El primer nodo es el maestro**; los demás son sus subordinados. Los enlaces resultantes quedan **fijados a 1 y protegidos** (*golden genes*): ni el QEA los rota ni el GA los muta.

```
Restricción 1 (nodos separados por espacio): 1 2 3
  ✓ Restricción 1: nodo 1 es maestro de [2, 3]
Restricción 2 (nodos separados por espacio): 4 5 6
  ✓ Restricción 2: nodo 4 es maestro de [5, 6]
Restricción 3 (nodos separados por espacio):      ← Enter vacío para terminar
```

Los nodos se numeran **desde 1**, no desde 0.

### Salida

El script ejecuta cuatro fases y produce:

1. Traza de convergencia del QEA en consola (cada 25 generaciones)
2. Traza de convergencia del GA
3. **`resultados_integrado.png`** — figura de 6 paneles: convergencia, diversidad, topologías óptimas de QEA y GA, y tabla comparativa
4. Análisis estadístico: media ± desviación de ambos algoritmos y **prueba de Wilcoxon** (α = 0.05, hipótesis alternativa `QEA < GA`)

---

## Uso programático

Si prefieres saltarte el diálogo interactivo e integrar el código en tus propios experimentos:

```python
import numpy as np
from qea_rover_integrado_mejorado import (
    ExperimentConfig,
    ChromosomeEvaluator,
    QEA,
    GeneticAlgorithm,
    plot_comparison,
    run_statistical_analysis,
)

n = 8

# 1. Matriz de costos de comunicación (MCC), simétrica
rng = np.random.default_rng(42)
cost_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        cost_matrix[i, j] = cost_matrix[j, i] = round(rng.uniform(1.0, 6.0), 3)

# 2. Configuración compartida
cfg = ExperimentConfig(
    n_agents=n,
    constraints=[[1, 2, 3], [4, 5, 6]],  # nodo 1 maestro de 2,3 — nodo 4 de 5,6
    max_generations=150,
    rotation_scheme="I",
    use_qiskit=False,  # True para simulación con Qiskit
    seed=42,
)

# 3. Evaluador compartido: única fuente de verdad para ambos algoritmos
ev = ChromosomeEvaluator(cfg.n_agents, cost_matrix, cfg.constraints)

# 4. Ejecutar
qea = QEA(cfg, ev).run(verbose=True)
ga = GeneticAlgorithm(cfg, ev).run(verbose=True)

print(f"QEA: {qea.best_fitness:.4f}   GA: {ga.best_fitness:.4f}")

# 5. Figura y estadística (opcionales)
plot_comparison(qea, ga, cfg, ev, save_path="mi_resultado.png")
stats = run_statistical_analysis(cfg, cost_matrix, n_replicas=30)
```

Ambos algoritmos devuelven un `AlgorithmResult` con: `best_binary`, `best_fitness`, `fitness_history`, `best_history`, `diversity_history`, `time_elapsed` y `label`.

---

## Cómo funciona

### Codificación

Una topología de `n` agentes se representa como un vector binario de `n(n-1)/2` genes — el triángulo superior de la matriz de adyacencia. El gen que conecta los nodos `i < j` (base 1) está en el índice:

```
idx = (2n − i)(i − 1)/2 + (j − i) − 1
```

`1` = enlace de comunicación presente, `0` = ausente.

### Función de aptitud

`Chromosome.get_total_cost()` implementa la Ecuación 1 de Carvajal-Godínez:

```
CI = Σᵢ Σⱼ  c_ij · A_ij(x)
```

Es un problema de **minimización**: menor CI es mejor.

### Genes protegidos (*golden genes*)

Dos fuentes los generan:

- **Por defecto**, los primeros `n−1` genes valen 1 — el nodo 1 se conecta a todos los demás, garantizando conectividad base.
- **Las restricciones** que definas fijan a 1 los enlaces maestro↔subordinado.

`ChromosomeEvaluator` expone estos índices vía `get_protected_indices()` / `get_free_indices()`. El QEA bloquea esos qubits en β=1 (nunca los rota) y el GA los excluye de la mutación, así que ambos exploran el mismo subespacio.

### Esquemas de rotación del QEA

| Esquema | Condición para rotar | Carácter |
|---|---|---|
| **I** | bits difieren (rotación completa si además la solución es peor, θ×0.5 si no) | Expresivo, mejor exploración |
| **II** | bits difieren **y** solución peor | Conservador, converge más lento |
| **III** | bits difieren, sin importar el fitness | Agresivo, riesgo de convergencia prematura |

El ángulo decae exponencialmente (GDAA): `θ(g) = max(θ₀·e^(−λg), θ_min)`.

---

## Notas y limitaciones conocidas

- **`use_qiskit=True` es lento.** Construye y mide un circuito de `n(n-1)/2` qubits por generación, con 1 shot cada vez. Para `n > 15` el script recomienda automáticamente el modo clásico; el análisis estadístico siempre lo fuerza a `False`.
- **`scenario` es solo una etiqueta.** Se imprime y aparece en los títulos de la figura, pero no altera la matriz de costos ni las restricciones.
- **Métodos de coherencia no usados en el flujo principal.** `Chromosome.get_coherence()`, `test_degree_coord()`, `test_degree_funct()` y `get_fitness()` invocan `graph.degree().values()`, sintaxis de NetworkX 1.x que falla en NetworkX 2.0+. No afecta la ejecución porque `get_total_cost()` no depende del grafo, pero esos métodos no son utilizables tal cual con versiones modernas de NetworkX.
- **La MCC se genera aleatoriamente** a partir de la semilla. Para usar costos reales de tu misión, sustituye `cost_matrix` por tu propia matriz simétrica en el uso programático.
- **Reproducibilidad parcial.** `Chromosome.__init__` usa `random.random()` del módulo estándar para inicializar genes no protegidos, fuera del generador sembrado del experimento. Fija `random.seed(...)` si necesitas reproducibilidad estricta bit a bit.
- Matplotlib usa el backend `Agg` (sin ventana interactiva): la figura se guarda a disco, no se muestra en pantalla.

---

## Referencias

- Carvajal-Godínez, J. — modelo de costo de integración para arquitecturas MAS (Ec. 1)
- Xiong et al. (2018) — puertas de rotación cuántica en algoritmos evolutivos
