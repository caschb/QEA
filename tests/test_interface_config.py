"""Regression checks for configuration routing and algorithm execution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qea import (
    ChromosomeEvaluator,
    EvolutionaryAlgorithm,
    ExperimentConfig,
    QEAConfig,
    RunContext,
    create_algorithm,
    generate_mcc,
    read_config,
)
from qea.registry import load_parameters
from qea.validation import ConfigError


class InterfaceConfigTests(unittest.TestCase):
    def read_text(self, text: str) -> ExperimentConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(text)
            return read_config(path)

    def test_example_matches_defaults(self) -> None:
        self.assertEqual(read_config(Path("config.toml")), read_config(None))

    def test_sections_route_parameters(self) -> None:
        cfg = self.read_text("""
[problem]
n_agents = 4
constraints = [[1, 2]]
[experiment]
seed = 17
max_generations = 5
n_replicas = 3
[algorithms.qea]
use_qiskit = false
rotation_scheme = "III"
[algorithms.ga]
pop_size = 7
mutation_rate = 0.3
""")
        self.assertEqual(cfg.seed, 17)
        self.assertEqual(cfg.constraints, [[1, 2]])
        self.assertEqual(cfg.algorithms["qea"].rotation_scheme, "III")
        self.assertEqual(cfg.algorithms["ga"].pop_size, 7)
        self.assertFalse(hasattr(cfg.algorithms["ga"], "rotation_scheme"))
        self.assertFalse(hasattr(cfg.algorithms["qea"], "pop_size"))

    def test_rejects_unknown_misplaced_and_invalid_values(self) -> None:
        cases = (
            "seed = 42",
            "[problem]\nseed = 42",
            "[experiment]\nseed = true",
            "[algorithms.other]\nx = 1",
            "[algorithms.qea]\npop_size = 3",
            "[algorithms.ga]\nmutation_rate = 2.0",
            "[algorithms.qea]\ntheta_min = 1.0",
            'problem = "bad"',
            "algorithms = []",
            "[algorithms]\nga = 1",
        )
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ConfigError):
                self.read_text(text)

    def test_classical_config_does_not_query_aer(self) -> None:
        with patch("qea.algorithms.qea._aer_methods", side_effect=AssertionError):
            cfg = self.read_text("[algorithms.qea]\nuse_qiskit = false")
        self.assertFalse(cfg.algorithms["qea"].use_qiskit)

    def test_factory_rejects_wrong_parameter_type(self) -> None:
        with self.assertRaises(ConfigError):
            create_algorithm("ga", QEAConfig(use_qiskit=False))

    def test_seeded_results_preserve_behavior_and_reset_state(self) -> None:
        baseline = json.loads(
            (Path(__file__).parent / "fixtures/seeded_results.json").read_text(),
        )
        ev = ChromosomeEvaluator(4, generate_mcc(42, 4), [[1, 2, 3]])
        context = RunContext(ev, 42, 5, verbose=False)
        parameters = load_parameters(
            {"qea": {"use_qiskit": False}, "ga": {"pop_size": 4}},
        )
        for name, label in (("qea", "QEA"), ("ga", "GeneticAlgorithm")):
            algorithm = create_algorithm(name, parameters[name])
            self.assertIsInstance(algorithm, EvolutionaryAlgorithm)
            for _ in range(2):
                result = algorithm.run(context)
                values = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in vars(result).items()
                    if k != "time_elapsed"
                }
                self.assertEqual(values, baseline[label])
                self.assertTrue(
                    (result.best_binary[ev.get_protected_indices()] == 1).all(),
                )

    def test_context_rejects_invalid_budget(self) -> None:
        ev = ChromosomeEvaluator(3, generate_mcc(42, 3))
        with self.assertRaises(ConfigError):
            RunContext(ev, max_generations=0)


if __name__ == "__main__":
    unittest.main()
