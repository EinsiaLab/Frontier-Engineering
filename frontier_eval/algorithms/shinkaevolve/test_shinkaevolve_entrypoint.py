from __future__ import annotations

import unittest

from frontier_eval.algorithms.shinkaevolve import shinkaevolve_entrypoint as entrypoint


class TestShinkaEvolveEntrypoint(unittest.TestCase):
    def test_extract_metrics_and_artifacts_from_nested_dict(self) -> None:
        metrics, artifacts = entrypoint._extract_metrics_and_artifacts(
            {
                "metrics": {"combined_score": -1.0, "valid": 0.0},
                "artifacts": {"error_message": "boom"},
            }
        )

        self.assertEqual(metrics["combined_score"], -1.0)
        self.assertEqual(artifacts["error_message"], "boom")

    def test_primary_error_message_reads_nested_unified_artifact(self) -> None:
        error = entrypoint._primary_error_message(
            {
                "user_artifact::error_message": "candidate infeasible on case 2",
            }
        )

        self.assertEqual(error, "candidate infeasible on case 2")

    def test_synthesize_text_feedback_prioritizes_error_and_runtime_problem(self) -> None:
        feedback = entrypoint._synthesize_text_feedback(
            {"combined_score": -1e18, "valid": 0.0, "benchmark_returncode": 0.0},
            {
                "user_artifact::error_message": "Traceback: KeyError: 'wind_u'",
                "agent_files": "\n".join(
                    [
                        "Task.md",
                        "README.md",
                        "baseline/solution.py",
                        "runtime/problem.py",
                    ]
                ),
                "constraints": "Edit only scripts/init.py.",
                "agent_file::Task.md": "Task contract",
                "agent_file::README.md": "README content",
                "agent_file::baseline/solution.py": "def solve(instance): return baseline()",
                "agent_file::runtime/problem.py": "INSTANCE_KEYS = ['time_grid', 'weather_cube']",
                "benchmark_stdout": '{"combined_score": -1e18, "valid": 0.0}',
            },
        )

        self.assertIn("## Error Message", feedback)
        self.assertIn("KeyError: 'wind_u'", feedback)
        self.assertIn("## Agent File: runtime/problem.py", feedback)
        self.assertIn("INSTANCE_KEYS", feedback)
        self.assertLess(feedback.index("## Error Message"), feedback.index("## Constraint Summary"))
        self.assertLess(feedback.index("## Agent File: runtime/problem.py"), feedback.index("## Benchmark Stdout"))


if __name__ == "__main__":
    unittest.main()
