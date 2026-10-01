"""Comparison plots for a named collection of algorithm results."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .evaluator import ChromosomeEvaluator
from .result import AlgorithmResult


def plot_comparison(
    results: Mapping[str, AlgorithmResult],
    cfg: ExperimentConfig,
    evaluator: ChromosomeEvaluator,
    save_path: str = "resultados_integrado.png",
) -> None:
    """Plot comparable best costs and separate algorithm-specific diagnostics.

    Histories may have different lengths or be absent. Registry names are the
    identity; result labels need not be unique. Diagnostic scales are separate
    because algorithms currently use different definitions of diversity/fitness.
    """
    if not results:
        raise ValueError("At least one algorithm result is required")
    bg, grid_c, txt = "#0d1b2a", "#1e3a4a", "#c8d8e8"
    fig = plt.figure(figsize=(16, 6 + 3.3 * len(results)), facecolor="#0a0f18")
    grid = fig.add_gridspec(
        len(results) + 2,
        3,
        height_ratios=[1.2, max(0.6, 0.24 * (len(results) + 1))] + [1] * len(results),
    )

    def style(ax, title):
        ax.set_facecolor(bg)
        ax.tick_params(colors=txt)
        ax.set_title(title, color=txt, fontsize=10)
        ax.grid(True, color=grid_c, alpha=0.5)
        ax.xaxis.label.set_color(txt)
        ax.yaxis.label.set_color(txt)
        for spine in ax.spines.values():
            spine.set_color(grid_c)

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
                plt.Circle(
                    (x, y), 0.13, color="#0d1b2a", ec=node_color, lw=2, zorder=2
                ),
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

    try:
        convergence = fig.add_subplot(grid[0, :])
        palette = plt.get_cmap("tab20")
        for index, (name, result) in enumerate(results.items()):
            convergence.plot(
                range(len(result.best_history)),
                result.best_history,
                label=name,
                marker="o" if len(result.best_history) == 1 else None,
                color=palette(index % 20),
            )
        style(convergence, "Convergencia — mejor costo encontrado (menor es mejor)")
        convergence.set_xlabel("Generación")
        convergence.set_ylabel("Costo CI")
        convergence.legend(facecolor=bg, labelcolor=txt)

        metrics = fig.add_subplot(grid[1, :])
        metrics.axis("off")
        table = metrics.table(
            cellText=[
                [name, f"{r.best_fitness:.3f}", f"{r.time_elapsed:.2f}"]
                for name, r in results.items()
            ],
            colLabels=["Algoritmo", "Mejor CI", "Tiempo (s)"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        for cell in table.get_celld().values():
            cell.set_facecolor(bg)
            cell.set_edgecolor(grid_c)
            cell.get_text().set_color(txt)

        for index, (name, result) in enumerate(results.items()):
            row = index + 2
            topology = fig.add_subplot(grid[row, 0])
            draw_topo(
                topology,
                result.best_binary,
                palette(index % 20),
                f"{name} — topología | CI = {result.best_fitness:.3f}",
            )
            for column, values, title in (
                (1, result.fitness_history, "Aptitud interna"),
                (2, result.diversity_history, "Diversidad interna"),
            ):
                ax = fig.add_subplot(grid[row, column])
                ax.plot(range(len(values)), values, color=palette(index % 20))
                style(ax, f"{name} — {title} (escala propia)")
                ax.set_xlabel("Generación")
                if not values:
                    ax.text(
                        0.5,
                        0.5,
                        "Sin diagnóstico",
                        color=txt,
                        transform=ax.transAxes,
                        ha="center",
                    )
        fig.suptitle(
            f"Comparación de algoritmos — {cfg.n_agents} agentes | {cfg.scenario}\n"
            f"Evaluación compartida: Chromosome.get_total_cost() | "
            f"Genes protegidos: {len(evaluator.get_protected_indices())}",
            color=txt,
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0f18")
    finally:
        plt.close(fig)
    print(f"  ✓ Figura guardada: {save_path}")
