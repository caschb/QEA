# QEA vs GA — Optimización de topologías MAS para arquitecturas de rover

Comparación entre un **Algoritmo Evolutivo Cuántico (QEA)** y un **Algoritmo Genético (GA)** clásico para encontrar la topología de comunicación de menor costo en un sistema multi-agente (MAS), sujeta a restricciones de organización (equipos y jerarquías).

Ambos algoritmos comparten **exactamente la misma función de aptitud** — `Chromosome.get_total_cost()` — a través de un evaluador puente, de modo que la comparación es justa: mismo espacio de búsqueda, mismas restricciones, mismo criterio de costo.

> Beca CeNAT-CONARE | ITCR — CNCA
> Autor: Jonathan D. Fuentes Calvo · Tutor: Dr. Ing. Johan Carvajal Godínez

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `src/qea/chromosome.py` | Codificación de topologías, restricciones y función de costo. |
| `src/qea/qea_rover_integrado_mejorado.py` | QEA, GA, análisis estadístico y visualización. |
| `src/qea/config.py` | Configuración y validación. |
| `config.toml` | Ejemplo completo de configuración de la CLI. |

---

## Requisitos

- Python 3.13 o superior

```bash
uv sync --locked
```

`qiskit` y `qiskit-aer` solo son necesarios si se usa el modo de observación cuántica real. El script también ofrece un modo clásico (muestreo pseudoaleatorio) que reproduce el colapso del Q-bit sin simulador.

---

## Ejecución desde la CLI (incluido en un cluster)

```bash
git clone https://github.com/johan-carvajal-godinez/QEA.git
cd QEA
uv sync --locked
uv run --no-sync qea -c config.toml
```

La CLI no solicita entrada interactiva. `-c` acepta una ruta TOML; sin ella se usan los valores por defecto de `ExperimentConfig`, iguales a los de `config.toml`. Copia ese archivo para cada experimento y cambia sus claves. La validación rechaza claves desconocidas y valores inválidos antes de iniciar el cálculo.

| Grupo | Claves principales |
|---|---|
| Problema | `n_agents` (mínimo 3), `scenario` (`nominal`, `safe`, `critical`), `max_generations` (mínimo 1), `seed` (entero no negativo), `constraints` |
| QEA | `theta_initial`, `theta_min`, `decay_rate`, `rotation_scheme` (`I`, `II`, `III`), `use_qiskit`, `aer_method` |
| GA | `pop_size` (mínimo 3), `mutation_rate` y `crossover_rate` (entre 0 y 1) |
| Estadística | `n_replicas` (mínimo 2) |

`config.toml` documenta cada parámetro y sus valores de ejemplo. `constraints` es una lista de equipos, por ejemplo `[[1, 2, 3], [4, 5, 6]]`. El primer nodo de cada equipo es el maestro; los restantes son subordinados. Los nodos empiezan en 1 y cada equipo necesita al menos dos nodos distintos dentro de `1..n_agents`.

Para una prueba rápida, usa una copia del TOML con `use_qiskit = false`, `max_generations = 2` y `n_replicas = 2`. Qiskit puede ser mucho más lento. Las réplicas estadísticas siempre usan el modo clásico y como máximo 50 generaciones.

Para un trabajo por lotes, instala las dependencias antes de enviarlo y ejecuta desde un directorio de resultados. Usa rutas absolutas en el script del planificador:

```bash
cd /ruta/a/resultados/experimento-01
/ruta/a/QEA/.venv/bin/qea -c /ruta/a/QEA/config.toml > run.log 2>&1
```

La ejecución imprime el progreso y el resumen en la salida estándar. Guarda `resultados_integrado.png` en el directorio de trabajo; otra ejecución allí sobrescribe la figura. La matriz de costos de comunicación se genera a partir de `seed`; la CLI todavía no acepta una matriz externa. `scenario` solo etiqueta el experimento.

---

## Uso programático

Si prefieres saltarte el diálogo interactivo e integrar el código en tus propios experimentos:

```python
import numpy as np
from qea import (
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
