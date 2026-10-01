"""Paired repetitions and statistical comparisons of selected algorithms."""

from itertools import combinations
from typing import TypedDict

import numpy as np
from scipy import stats

from .config import ExperimentConfig
from .evaluator import ChromosomeEvaluator
from .execution import run_algorithms


class PairComparison(TypedDict):
    left: str
    right: str
    statistic: float
    p_value: float
    adjusted_p_value: float
    significant: bool


def compare_scores(scores: dict[str, list[float]]) -> list[PairComparison]:
    """Two-sided paired Wilcoxon tests with Holm family-wise correction.

    Identical paired scores yield p=1 without calling Wilcoxon, whose
    zero-difference case can otherwise produce warnings or NaN values.
    """
    comparisons: list[PairComparison] = []
    for left, right in combinations(scores, 2):
        differences = np.asarray(scores[left]) - np.asarray(scores[right])
        if np.all(differences == 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(scores[left], scores[right], alternative="two-sided")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        comparisons.append(
            {
                "left": left,
                "right": right,
                "statistic": statistic,
                "p_value": p_value,
                "adjusted_p_value": 1.0,
                "significant": False,
            }
        )
    adjusted = 0.0
    for rank, pair in enumerate(sorted(comparisons, key=lambda p: p["p_value"])):
        adjusted = max(adjusted, min(1.0, pair["p_value"] * (len(comparisons) - rank)))
        pair["adjusted_p_value"] = adjusted
        pair["significant"] = adjusted < 0.05
    return comparisons


def run_statistical_analysis(
    cfg: ExperimentConfig,
    cost_matrix: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Repeat the configured experiment, changing only each repetition's seed.

    All algorithms in a repetition use the same problem instance. Parameters,
    backend and generation budget match the main run; no hidden overrides.
    """
    scores = {name: [] for name in cfg.selected_algorithms}
    times = {name: [] for name in cfg.selected_algorithms}
    for rep in range(cfg.n_replicas):
        evaluator = ChromosomeEvaluator(cfg.n_agents, cost_matrix, cfg.constraints)
        results = run_algorithms(
            cfg, evaluator, seed=cfg.seed + rep * 137, verbose=False
        )
        for name, result in results.items():
            scores[name].append(result.best_fitness)
            times[name].append(result.time_elapsed)
        if verbose:
            summary = " | ".join(
                f"{name}: {r.best_fitness:.3f}" for name, r in results.items()
            )
            print(f"  Rep {rep + 1}/{cfg.n_replicas} | {summary}")
    summaries = {
        name: {
            "scores": values,
            "times": times[name],
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "best": float(np.min(values)),
            "mean_time": float(np.mean(times[name])),
        }
        for name, values in scores.items()
    }
    comparisons = compare_scores(scores)
    if verbose:
        for name, summary in summaries.items():
            print(f"  {name}: μ={summary['mean']:.3f} ± {summary['std']:.3f}")
    return {"algorithms": summaries, "comparisons": comparisons}
