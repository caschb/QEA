"""End-to-end coverage of selection, extension, plots and paired analysis."""

import contextlib
import io
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qea import (
    ChromosomeEvaluator,
    ExperimentConfig,
    GAConfig,
    QEAConfig,
    RunContext,
    generate_mcc,
    plot_comparison,
    read_config,
    run_algorithms,
    run_statistical_analysis,
)
from qea.analysis import compare_scores
from qea.registry import ALGORITHMS, AlgorithmDefinition
from qea.result import AlgorithmResult
from qea.validation import ConfigError


@dataclass(frozen=True)
class StubConfig:
    offset: float = 0.0


class StubAlgorithm:
    contexts: ClassVar[list[RunContext]] = []

    def __init__(self, parameters: StubConfig) -> None:
        self.parameters = parameters

    def run(self, context: RunContext) -> AlgorithmResult:
        self.contexts.append(context)
        binary = context.evaluator.enforce_golden(
            np.zeros(context.evaluator.n_genes, dtype=int),
        )
        cost = context.evaluator.evaluate(binary) + self.parameters.offset
        return AlgorithmResult(
            binary,
            cost,
            best_history=[cost],
            label="duplicate label",
        )


class ExecutionAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mcc = generate_mcc(42, 4)
        self.evaluator = ChromosomeEvaluator(4, self.mcc, [[1, 2]])
        StubAlgorithm.contexts = []

    def config(self, names: tuple[str, ...] = ("qea", "ga")) -> ExperimentConfig:
        return ExperimentConfig(
            n_agents=4,
            max_generations=3,
            n_replicas=3,
            constraints=[[1, 2]],
            selected_algorithms=names,
            algorithms={"qea": QEAConfig(use_qiskit=False), "ga": GAConfig(pop_size=4)},
        )

    def test_selection_and_order(self) -> None:
        for names in (("ga",), ("ga", "qea")):
            results = run_algorithms(self.config(names), self.evaluator, verbose=False)
            self.assertEqual(list(results), list(names))

    def test_selection_validation(self) -> None:
        for selection in ("[]", '["ga", "ga"]', '["missing"]', "[true]", '"ga"'):
            with (
                self.subTest(selection=selection),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "config.toml"
                path.write_text(f"[experiment]\nalgorithms = {selection}")
                with self.assertRaises(ConfigError):
                    read_config(path)

    def test_ga_only_does_not_construct_qea_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[experiment]\nalgorithms = ["ga"]')
            with patch("qea.algorithms.qea._aer_methods", side_effect=AssertionError):
                cfg = read_config(path)
            self.assertEqual(set(cfg.algorithms), {"ga"})

    def test_third_algorithm_without_runner_or_analysis_changes(self) -> None:
        with patch.dict(
            ALGORITHMS,
            {"stub": AlgorithmDefinition(StubConfig, StubAlgorithm)},
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text("""[problem]
n_agents = 4
constraints = [[1, 2]]
[experiment]
algorithms = ["ga", "stub", "qea"]
max_generations = 3
n_replicas = 3
[algorithms.qea]
use_qiskit = false
[algorithms.stub]
offset = 0.25
""")
                cfg = read_config(path)
            results = run_algorithms(cfg, self.evaluator, verbose=False)
            analysis = run_statistical_analysis(cfg, self.mcc, verbose=False)
            self.assertEqual(list(results), ["ga", "stub", "qea"])
            self.assertEqual(len(analysis["comparisons"]), 3)
            self.assertEqual(len(analysis["algorithms"]["stub"]["scores"]), 3)
            self.assertEqual(
                [c.seed for c in StubAlgorithm.contexts],
                [42, 42, 179, 316],
            )
            self.assertTrue(
                all(
                    c.max_generations == cfg.max_generations
                    for c in StubAlgorithm.contexts
                )
            )
            self.assertEqual(StubAlgorithm.contexts[0].evaluator, self.evaluator)
            self.check_plot(results, cfg)

    def test_single_algorithm_statistics_and_plot(self) -> None:
        cfg = self.config(("ga",))
        analysis = run_statistical_analysis(cfg, self.mcc, verbose=False)
        self.assertEqual(analysis["comparisons"], [])
        self.check_plot(run_algorithms(cfg, self.evaluator, verbose=False), cfg)

    def check_plot(
        self, results: dict[str, AlgorithmResult], cfg: ExperimentConfig
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            path = Path(directory) / "plot.png"
            before = plt.get_fignums()
            plot_comparison(results, cfg, self.evaluator, str(path))
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(plt.get_fignums(), before)

    def test_identical_scores_are_not_significant(self) -> None:
        with patch("qea.analysis.stats.wilcoxon", side_effect=AssertionError):
            pair = compare_scores({"a": [1, 1], "b": [1, 1]})[0]
        self.assertEqual(pair["p_value"], 1.0)
        self.assertFalse(pair["significant"])

    def test_holm_correction_and_two_sided_tests(self) -> None:
        with patch(
            "qea.analysis.stats.wilcoxon",
            side_effect=[
                SimpleNamespace(statistic=0, pvalue=0.01),
                SimpleNamespace(statistic=0, pvalue=0.04),
                SimpleNamespace(statistic=0, pvalue=0.03),
            ],
        ) as wilcoxon:
            pairs = compare_scores({"a": [1, 2], "b": [2, 3], "c": [4, 5]})
        self.assertEqual([p["adjusted_p_value"] for p in pairs], [0.03, 0.06, 0.06])
        self.assertEqual([p["significant"] for p in pairs], [True, False, False])
        self.assertTrue(
            all(
                c.kwargs["alternative"] == "two-sided" for c in wilcoxon.call_args_list
            ),
        )

    def test_statistics_preserve_quantum_parameters_and_budget(self) -> None:
        cfg = ExperimentConfig(
            n_agents=4,
            max_generations=51,
            n_replicas=2,
            selected_algorithms=("qea",),
        )
        contexts: ClassVar[list[RunContext]] = []

        def run(context: RunContext) -> AlgorithmResult:
            contexts.append(context)
            return AlgorithmResult(np.zeros(6, dtype=int), 1.0)

        with patch(
            "qea.execution.create_algorithm",
            return_value=SimpleNamespace(run=run),
        ) as factory:
            run_statistical_analysis(cfg, self.mcc, verbose=False)
        self.assertTrue(all(c.max_generations == cfg.max_generations for c in contexts))
        self.assertIs(factory.call_args.args[1], cfg.algorithms["qea"])
        self.assertTrue(factory.call_args.args[1].use_qiskit)


if __name__ == "__main__":
    unittest.main()
