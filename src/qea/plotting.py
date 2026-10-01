"""Plots comparing QEA and GA results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from .config import ExperimentConfig
from .evaluator import ChromosomeEvaluator
from .result import AlgorithmResult


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
