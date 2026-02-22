from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


class TestHighReliableSimulationEvaluator(unittest.TestCase):
    def test_init_program_can_be_evaluated(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        eval_path = (
            repo
            / "benchmarks"
            / "WirelessChannelSimulation"
            / "HighReliableSimulation"
            / "eval"
            / "evaluator.py"
        )
        program_path = (
            repo
            / "benchmarks"
            / "WirelessChannelSimulation"
            / "HighReliableSimulation"
            / "scripts"
            / "init.py"
        )

        spec = importlib.util.spec_from_file_location("hrs_eval", str(eval_path))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.evaluate(str(program_path), repo_root=repo)
        metrics = result.metrics if hasattr(result, "metrics") else result

        required_keys = {
            "combined_score",
            "runtime_s",
            "error_log_ratio",
            "valid",
            "err_rate_log_median",
            "actual_std_median",
            "runtime_s_total",
        }
        self.assertTrue(required_keys.issubset(metrics.keys()))
        self.assertGreater(metrics["runtime_s_total"], 0.0)
        self.assertIn(metrics["valid"], (0.0, 1.0))
        self.assertGreaterEqual(metrics["combined_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
